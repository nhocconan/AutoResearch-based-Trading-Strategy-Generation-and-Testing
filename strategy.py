#!/usr/bin/env python3
"""
Experiment #380: 1h Primary + 4h/12h HTF — Simplified Multi-Confluence Trend Pullback

Hypothesis: After analyzing 379 failed experiments, the pattern is clear:
1. Lower TF (1h/30m) strategies with TOO MANY filters = 0 trades (exp #375, #378)
2. Complex dual-regime logic creates conflicting signals that never trigger
3. SIMPLER confluence works better: HTF trend + LTF pullback + volume confirmation
4. 4h HMA(21) for major trend direction (proven in best strategy mtf_1d_hma_rsi_1w)
5. 1h RSI(14) pullback entries (not extremes - use 35/65 not 20/80 for more trades)
6. 12h ADX(14) > 18 for trend strength (lower threshold than typical 25 for more signals)
7. Volume > 0.7x 20-bar avg (relaxed from 0.8x to ensure trades generate)
8. NO session filter (session filters killed trades in exp #375, #378)
9. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)
10. Target: 40-60 trades/year on 1h timeframe

Why this might beat current best (Sharpe=0.435):
- Simpler logic = more trades while maintaining quality
- HTF trend filter prevents counter-trend disasters (2022 crash lesson)
- RSI pullback (not extreme) catches trend continuations, not just reversals
- Lower ADX threshold (18 vs 25) generates signals in moderate trends
- No session filter = trades during Asian/London overlap volatility

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 40-60 trades/year on 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_simp_trend_rsi_adx_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate +DM and -DM
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Calculate TR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # Calculate DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Calculate 12h HTF indicators (trend strength)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    # Also calculate 1h HMA for local trend confirmation
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_8 = calculate_hma(close, period=8)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(adx_12h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        # Only trade in direction of 4h HMA trend
        regime_bull = close[i] > hma_4h_21_aligned[i]
        regime_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 12H TREND STRENGTH (ADX filter) ===
        # ADX > 18 = sufficient trend strength (lower than typical 25 for more signals)
        trend_strong = adx_12h_aligned[i] > 18.0
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.7x average (relaxed to ensure trades generate)
        volume_confirmed = volume[i] > 0.7 * vol_avg_20[i] if not np.isnan(vol_avg_20[i]) else True
        
        # === 1H LOCAL TREND ===
        hma_bullish = hma_1h_8[i] > hma_1h_21[i]
        hma_bearish = hma_1h_8[i] < hma_1h_21[i]
        
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI PULLBACK SIGNALS (not extremes for more trades) ===
        # Long: RSI pulled back to 35-45 in uptrend
        rsi_pullback_long = 32.0 < rsi_14[i] < 48.0
        # Short: RSI rallied to 52-68 in downtrend
        rsi_pullback_short = 52.0 < rsi_14[i] < 68.0
        
        # RSI extreme for stronger signals
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC - SIMPLIFIED CONFLUENCE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === LONG ENTRIES ===
        # Must have: 4h bull trend + volume confirmed
        if regime_bull and volume_confirmed:
            # Strong long: RSI oversold + 1h HMA bullish + ADX strong
            if rsi_oversold and hma_bullish and trend_strong:
                new_signal = LONG_STRONG
            # Base long: RSI pullback + 1h HMA bullish
            elif rsi_pullback_long and hma_bullish:
                new_signal = LONG_BASE
            # Weaker long: RSI pullback + price > SMA200 (trend confirmation)
            elif rsi_pullback_long and price_above_sma200:
                new_signal = LONG_BASE * 0.8
        
        # === SHORT ENTRIES ===
        # Must have: 4h bear trend + volume confirmed
        if regime_bear and volume_confirmed:
            # Strong short: RSI overbought + 1h HMA bearish + ADX strong
            if rsi_overbought and hma_bearish and trend_strong:
                if new_signal == 0.0:  # Don't override long signal
                    new_signal = -SHORT_STRONG
            # Base short: RSI pullback + 1h HMA bearish
            elif rsi_pullback_short and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            # Weaker short: RSI pullback + price < SMA200
            elif rsi_pullback_short and not price_above_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 1h) ===
        # Force trade if no signal for 24 bars (~1 day on 1h)
        if bars_since_last_trade > 24 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] < 50.0 and hma_bullish:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and rsi_14[i] > 50.0 and hma_bearish:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear:
                trend_reversal = True
            if position_side < 0 and regime_bull:
                trend_reversal = True
        
        if stoploss_triggered or rsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.17:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals