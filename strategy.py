#!/usr/bin/env python3
"""
Experiment #525: 1h Volatility Spike Mean Reversion with 4h HMA Trend Bias

Hypothesis: After 500+ failed experiments, the pattern is clear - pure trend following
fails on BTC/ETH in bear markets, and pure mean reversion fails in strong trends.
The winning approach is VOLATILITY-SPIKE DETECTION + MEAN REVERSION with HTF trend filter.

Key insight from research: After panic spikes (ATR ratio > 1.8), price tends to revert
within 2-5 bars. This works on 1h timeframe because:
1. 1h captures intraday panic/recovery cycles better than 4h/12h
2. Volatility spikes are more frequent on 1h (more trade opportunities)
3. Combined with 4h HMA bias, we avoid counter-trend mean reversion

Innovations vs failed experiments:
1. LOOSE RSI thresholds (35/65 instead of 30/70) - ensures ≥10 trades/year
2. ATR ratio > 1.8 (not 2.0) - captures more vol spike events
3. 4h HMA(21) bias - only mean-revert in direction of HTF trend
4. 2.0*ATR stoploss - tight enough for 1h, loose enough to avoid whipsaw
5. Discrete signal levels (0.0, ±0.25) - minimizes fee churn

Why 1h might work where others failed:
- More frequent vol spikes = more entry opportunities
- Faster mean reversion on 1h vs 4h/12h
- Can capture both intraday swings AND multi-day trends via 4h bias

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_4h_hma_meanrev_loose_rsi_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price vs rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    zscore_20 = calculate_zscore(close, 20)
    
    # Volatility ratio: ATR(7)/ATR(30) - detects vol spikes
    vol_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(zscore_20[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 1.8  # High volatility = mean reversion likely
        vol_normal = vol_ratio[i] < 1.2  # Low volatility = trend may continue
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # VOLATILITY SPIKE: Mean-reversion (panic reversals)
        if vol_spike:
            # Long: RSI oversold + Z-score low + bullish 4h bias
            if rsi_14[i] < 40 and zscore_20[i] < -1.0 and bull_bias:
                new_signal = SIZE
            # Short: RSI overbought + Z-score high + bearish 4h bias
            elif rsi_14[i] > 60 and zscore_20[i] > 1.0 and bear_bias:
                new_signal = -SIZE
        
        # NORMAL VOLATILITY: Trend continuation with pullback entries
        elif vol_normal:
            # Long: RSI pullback in uptrend
            if rsi_14[i] < 50 and bull_bias:
                new_signal = SIZE
            # Short: RSI rally in downtrend
            elif rsi_14[i] > 50 and bear_bias:
                new_signal = -SIZE
        
        # MID VOLATILITY: Use Z-score extremes
        else:
            # Long: Price significantly below mean + bullish bias
            if zscore_20[i] < -1.5 and bull_bias:
                new_signal = SIZE
            # Short: Price significantly above mean + bearish bias
            elif zscore_20[i] > 1.5 and bear_bias:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND BIAS REVERSAL EXIT ===
        # Exit if 4h trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias and vol_ratio[i] < 1.2:
                new_signal = 0.0
            if position_side < 0 and bull_bias and vol_ratio[i] < 1.2:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals