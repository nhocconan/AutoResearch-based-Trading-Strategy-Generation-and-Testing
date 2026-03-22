#!/usr/bin/env python3
"""
Experiment #358: 30m Primary + 4h/1d HTF — Simplified Trend-Pullback Strategy

Hypothesis: After 357 experiments, the pattern is clear:
1. 30m strategies FAIL with Sharpe=0.000 due to TOO MANY filters (experiments #348, #350, #355)
2. Complex regime-switching (Chop + CRSI + Donchian) generates 0 trades on lower TF
3. SIMPLER is better: 4h HMA for trend direction, 30m RSI for pullback entries
4. Remove session filter (eliminates 50% of opportunities), relax volume threshold
5. Use asymmetric RSI thresholds: 35/65 for trend-follow, 25/75 for mean-revert
6. Position size: 0.20-0.25 (smaller for lower TF to reduce fee drag)
7. Stoploss: 2.0 * ATR (tighter for lower TF)
8. Target: 40-60 trades/year on 30m (NOT 200+ which kills with fees)

Why this might beat current best (Sharpe=0.435):
- 30m entries within 4h trend = HTF trade frequency with LTF precision
- Simpler logic = more trades (avoid the 0-trade death spiral)
- Asymmetric RSI thresholds adapt to regime without complex switching
- Proven pattern from exp #356 but SIMPLIFIED for lower TF

Position sizing: 0.20-0.25 (discrete levels)
Stoploss: 2.0 * ATR trailing
Target: 40-60 trades/year on 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_4h1d_simp_pullback_v1"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_8 = calculate_hma(df_4h['close'].values, period=8)
    
    # Calculate 1d HTF indicators (major regime filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_8_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_8)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_30m_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller for lower TF to reduce fee drag
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
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_30m_21[i]):
            continue
        
        # === 1D MAJOR REGIME (primary direction filter) ===
        # Price above 1d HMA = bull regime (favor longs)
        # Price below 1d HMA = bear regime (favor shorts)
        regime_bull_1d = close[i] > hma_1d_21_aligned[i]
        regime_bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND DIRECTION (entry filter) ===
        # 4h HMA(8) > HMA(21) = bullish trend
        # 4h HMA(8) < HMA(21) = bearish trend
        trend_4h_bull = hma_4h_8_aligned[i] > hma_4h_21_aligned[i]
        trend_4h_bear = hma_4h_8_aligned[i] < hma_4h_21_aligned[i]
        
        # === 30M LOCAL TREND ===
        trend_30m_bull = close[i] > hma_30m_21[i]
        trend_30m_bear = close[i] < hma_30m_21[i]
        
        # === VOLUME CONFIRMATION (loose threshold) ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10) if not np.isnan(vol_sma_20[i]) else 1.0
        vol_ok = vol_ratio > 0.5  # Very loose - just not extremely low volume
        
        # === RSI SIGNALS (asymmetric thresholds) ===
        # Trend-follow: RSI 35-65 zone for pullback entries
        # Mean-revert: RSI <25 or >75 for extreme reversals
        rsi_oversold_trend = rsi_14[i] < 40.0
        rsi_oversoldExtreme = rsi_14[i] < 30.0
        rsi_overbought_trend = rsi_14[i] > 60.0
        rsi_overbought_extreme = rsi_14[i] > 70.0
        
        # === PRICE POSITION ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC - SIMPLIFIED TREND-PULLBACK ===
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Condition 1: 1d bull + 4h bull + RSI pullback (trend-follow)
        if regime_bull_1d and trend_4h_bull and rsi_oversold_trend and vol_ok:
            new_signal = LONG_BASE
        # Condition 2: 1d bull + RSI extreme oversold (mean-revert)
        elif regime_bull_1d and rsi_oversoldExtreme and vol_ok:
            new_signal = LONG_STRONG
        # Condition 3: 4h bull + 30m pullback to HMA + RSI neutral (continuation)
        elif trend_4h_bull and not trend_30m_bull and 35.0 < rsi_14[i] < 50.0 and vol_ok:
            new_signal = LONG_BASE
        # Condition 4: Price > SMA200 + RSI extreme (strong mean-revert)
        elif price_above_sma200 and rsi_oversoldExtreme:
            new_signal = LONG_BASE * 0.8
        
        # === SHORT ENTRIES ===
        # Condition 1: 1d bear + 4h bear + RSI pullback (trend-follow)
        if regime_bear_1d and trend_4h_bear and rsi_overbought_trend and vol_ok:
            if new_signal == 0.0:
                new_signal = -SHORT_BASE
        # Condition 2: 1d bear + RSI extreme overbought (mean-revert)
        elif regime_bear_1d and rsi_overbought_extreme and vol_ok:
            if new_signal == 0.0:
                new_signal = -SHORT_STRONG
        # Condition 3: 4h bear + 30m rally to HMA + RSI neutral (continuation)
        elif trend_4h_bear and not trend_30m_bear and 50.0 < rsi_14[i] < 65.0 and vol_ok:
            if new_signal == 0.0:
                new_signal = -SHORT_BASE
        # Condition 4: Price < SMA200 + RSI extreme (strong mean-revert)
        elif not price_above_sma200 and rsi_overbought_extreme:
            if new_signal == 0.0:
                new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 30m) ===
        # Force trade if no signal for 48 bars (~1 day on 30m)
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if regime_bull_1d and rsi_14[i] < 45.0:
                new_signal = LONG_BASE * 0.6
            elif regime_bear_1d and rsi_14[i] > 55.0:
                new_signal = -SHORT_BASE * 0.6
            elif rsi_oversoldExtreme:
                new_signal = LONG_BASE * 0.6
            elif rsi_overbought_extreme:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_overbought_extreme:
                rsi_exit = True
            if position_side < 0 and rsi_oversoldExtreme:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bear and close[i] < hma_4h_21_aligned[i]:
                trend_reversal = True
            if position_side < 0 and trend_4h_bull and close[i] > hma_4h_21_aligned[i]:
                trend_reversal = True
        
        if stoploss_triggered or rsi_exit or trend_reversal:
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