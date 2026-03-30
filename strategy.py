#!/usr/bin/env python3
"""
Experiment #021: 4h Williams %R Reversal + Choppiness + Volume Confirmation

HYPOTHESIS: Williams %R hitting extreme readings after pullbacks gives high-probability
reversal entries. Combined with:
1. CHOP < 45 regime filter (only trend-following markets)
2. Volume confirmation (1.8x average = institutional interest)
3. 12h EMA(21) for HTF trend direction
4. Exit on opposite W%R extreme (not just stoploss)

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: Pullbacks to W%R < 10 reverse sharply due to uptrend strength
- Bear: W%R < 10 breakdowns lead to extended moves down (momentum breakdown)
- Range: CHOP > 61.8 filter prevents whipsaw entries in choppy markets

KEY DIFFERENCE FROM #015:
- #015 used Donchian breakout (laggy, catches moves late)
- #021 uses Williams %R extreme crossing (faster, captures reversal start)
- Target: 120-180 trades over 4 years (30-45/year)

RATIONALE FOR WILLIAMS %R:
- Unlike RSI, W%R uses highest high / lowest low in period = more sensitive
- < -80 = price at lowest point in lookback = potential reversal zone
- Range 5-15 = falling fast but not yet at extreme = better entry than waiting for < -80
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_reversal_chop_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """
    Williams %R
    < -80 = oversold (potential reversal)
    > -20 = overbought
    Entry: W%R crosses below -80 then recovers to 5-15 range = pullback entry
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            willr[i] = -100 * (highest - close[i]) / (highest - lowest)
    
    return willr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 45 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    willr = calculate_williams_r(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # 4h SMA(200) for trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR for stoploss
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 250  # 200 for SMA200 + 14 for W%R + 14 for CHOP + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr[i]) or np.isnan(chop[i]) or np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 45
        
        # Skip if choppy (key filter to avoid whipsaws)
        if is_choppy:
            if in_position:
                # Exit on choppy regime
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # === HTF TREND: 12h EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === LOCAL TREND: SMA(200) ===
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # === VOLUME CONFIRMATION (1.8x, proven threshold) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === WILLIAMS %R SIGNALS ===
        willr_value = willr[i]
        willr_prev = willr[i - 1] if i > 0 else -50
        
        # Detect W%R crossing below -80 (entering oversold)
        crossing_oversold = (willr_prev > -80) and (willr_value <= -80)
        
        # W%R in pullback zone: -80 < W%R < -5 (falling but not yet extreme)
        in_pullback = (-80 < willr_value < -5)
        
        # W%R recovery: crossed below -80 and now recovering toward -50
        w_cross_up = (willr_prev <= -80) and (willr_value > -80) and (willr_value > willr_prev)
        
        # W%R recovery zone: -50 < W%R < -5 (post-oversold recovery = entry)
        in_recovery = (-50 < willr_value < -5)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: W%R recovery in pullback + uptrend + volume ===
            # Entry: W%R crossed below -80 earlier, now recovering in -50 to -5 range
            # Confirms: uptrend (HTF + SMA200), volume spike, choppiness trending
            if htf_trend_up and above_sma200 and vol_spike and is_trending:
                # Check if W%R recently crossed oversold (within last 8 bars)
                recent_cross = False
                for back in range(1, min(8, i - warmup + 1)):
                    if i >= back:
                        prev_willr = willr[i - back]
                        prev_willr_1 = willr[i - back - 1] if (i - back - 1) >= warmup else -50
                        if (prev_willr_1 > -80) and (prev_willr <= -80):
                            recent_cross = True
                            break
                
                if recent_cross and in_recovery:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: W%R recovery to overbought + downtrend + volume ===
            # Entry: W%R crossed above -20 earlier, now falling back from +5 to +50 range
            if htf_trend_down and below_sma200 and vol_spike and is_trending:
                recent_cross_up = False
                for back in range(1, min(8, i - warmup + 1)):
                    if i >= back:
                        prev_willr = willr[i - back]
                        prev_willr_1 = willr[i - back - 1] if (i - back - 1) >= warmup else -50
                        if (prev_willr_1 < -20) and (prev_willr >= -20):
                            recent_cross_up = True
                            break
                
                if recent_cross_up and (-5 < willr_value < 50):
                    desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Long exit: W%R reaches overbought (> -20)
                if willr_value > -20:
                    desired_signal = 0.0
                
                # Stop if price falls 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short exit: W%R reaches oversold (< -80)
                if willr_value < -80:
                    desired_signal = 0.0
                
                # Stop if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals