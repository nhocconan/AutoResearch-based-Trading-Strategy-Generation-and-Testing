#!/usr/bin/env python3
"""
Experiment #348: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: After 30+ failed experiments with complex regime filters causing 0 trades,
return to PROVEN mechanics that generate consistent trades on lower timeframes:

1. 1d HMA(21) for MAJOR regime (bull/bear) - only trade in direction
2. 4h HMA(21) for INTERMEDIATE trend - entry direction filter
3. 30m RSI(14) pullback entries - buy dips in uptrend, sell rallies in downtrend
4. Session filter 8-20 UTC - reduces trades naturally (London/NY overlap)
5. Volume filter > 0.8x avg - avoid low liquidity entries
6. ATR(14) trailing stop 2.0x - tight stops for lower TF
7. Fallback entries every 15 bars - ensures minimum trade frequency

Why this might beat current best (Sharpe=0.435):
- 30m entries within 4h/1d trend = HTF win rate + LTF timing
- Session filter naturally limits to 40-60 trades/year (avoids fee drag)
- RSI pullback (not breakout) = better risk/reward in crypto
- Simpler logic than failed choppiness/regime strategies (338, 339, 340, 345)

Position sizing: 0.20-0.25 (smaller for 30m per Rule 10)
Stoploss: 2.0 * ATR trailing
Target: 40-60 trades/year on 30m (1 trade every 6-9 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_4h1d_session_vol_v1"
timeframe = "30m"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average for volume."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_30m_21 = calculate_hma(close, period=21)
    hma_30m_8 = calculate_hma(close, period=8)
    vol_sma_20 = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
    LONG_BASE = 0.20
    LONG_STRONG = 0.25
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_30m_21[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        vol_ok = vol_ratio > 0.8
        
        # === 1D MAJOR REGIME (primary direction filter) ===
        # Bull: price above 1d HMA (favor longs only)
        # Bear: price below 1d HMA (favor shorts only)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        trend_4h_bull = close[i] > hma_4h_21_aligned[i]
        trend_4h_bear = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope
        hma_4h_slope_up = hma_4h_21_aligned[i] > hma_4h_21_aligned[i-2] if i >= 2 else False
        hma_4h_slope_down = hma_4h_21_aligned[i] < hma_4h_21_aligned[i-2] if i >= 2 else False
        
        # === 30M LOCAL TREND ===
        hma_30m_bullish = hma_30m_8[i] > hma_30m_21[i]
        hma_30m_bearish = hma_30m_8[i] < hma_30m_21[i]
        
        # 30m HMA slope
        hma_30m_slope_up = hma_30m_21[i] > hma_30m_21[i-2] if i >= 2 else False
        hma_30m_slope_down = hma_30m_21[i] < hma_30m_21[i-2] if i >= 2 else False
        
        # === RSI PULLBACK SIGNALS (key entry logic) ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        
        # RSI momentum
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # RSI turning from extreme
        rsi_turning_long = rsi_14[i] > 35.0 and rsi_14[i-1] <= 35.0 if i > 0 else False
        rsi_turning_short = rsi_14[i] < 65.0 and rsi_14[i-1] >= 65.0 if i > 0 else False
        
        # === ENTRY LOGIC (simplified OR conditions for trade frequency) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require: session + volume + regime alignment)
        if in_session and vol_ok:
            if regime_bull:
                # Primary: 4h bull + 30m pullback + RSI turning up
                if trend_4h_bull and rsi_pullback_long and rsi_rising:
                    new_signal = LONG_BASE
                
                # Strong: 4h bull + 30m HMA bull + RSI > 40
                elif trend_4h_bull and hma_30m_bullish and rsi_14[i] > 40.0:
                    new_signal = LONG_STRONG
                
                # Fallback: regime bull + RSI turning from oversold
                elif rsi_turning_long and rsi_14[i] > 30.0:
                    if new_signal == 0.0:
                        new_signal = LONG_BASE * 0.8
            
            if regime_bear and trend_4h_bear:
                # Counter-trend long only if RSI very oversold
                if rsi_14[i] < 30.0 and rsi_rising:
                    if new_signal == 0.0:
                        new_signal = LONG_BASE * 0.5
        
        # SHORT ENTRIES (require: session + volume + regime alignment)
        if in_session and vol_ok:
            if regime_bear:
                # Primary: 4h bear + 30m pullback + RSI turning down
                if trend_4h_bear and rsi_pullback_short and rsi_falling:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
                
                # Strong: 4h bear + 30m HMA bear + RSI < 60
                elif trend_4h_bear and hma_30m_bearish and rsi_14[i] < 60.0:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG
                
                # Fallback: regime bear + RSI turning from overbought
                elif rsi_turning_short and rsi_14[i] < 70.0:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.8
            
            if regime_bull and trend_4h_bull:
                # Counter-trend short only if RSI very overbought
                if rsi_14[i] > 70.0 and rsi_falling:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.5
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 30m) ===
        # Force trade if no signal for 15 bars (~7.5 hours on 30m)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position and in_session:
            if regime_bull and trend_4h_bull and rsi_14[i] > 35.0:
                new_signal = LONG_BASE * 0.5
            elif regime_bear and trend_4h_bear and rsi_14[i] < 65.0:
                new_signal = -SHORT_BASE * 0.5
            elif rsi_14[i] < 28.0 and regime_bull:
                new_signal = LONG_BASE * 0.5
            elif rsi_14[i] > 72.0 and regime_bear:
                new_signal = -SHORT_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_14[i] > 70.0:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_14[i] < 30.0:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns bearish
            if position_side > 0 and regime_bear and close[i] < hma_1d_21_aligned[i]:
                regime_reversal = True
            # Short position but 1d regime turns bullish
            if position_side < 0 and regime_bull and close[i] > hma_1d_21_aligned[i]:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.23:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.18:
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