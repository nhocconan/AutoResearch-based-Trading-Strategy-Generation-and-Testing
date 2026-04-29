#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d EMA34 uptrend
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d EMA34 downtrend
# Volume confirmation reduces false signals. Works in bull/bear by following 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components on 6h data
    # EMA13 for power calculations
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34, 20)  # warmup for EMA13, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        # Using close vs EMA34 for regime determination
        is_bullish_regime = close[i] > curr_ema34
        is_bearish_regime = close[i] < curr_ema34
        
        # Elder Ray signals with confirmation
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Bull Power > 0 AND Bear Power rising (less negative than previous) AND bullish regime
                bear_power_rising = (i == start_idx) or (curr_bear_power > bear_power[i-1])
                if curr_bull_power > 0 and bear_power_rising and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 AND Bull Power falling (less positive than previous) AND bearish regime
                elif i > start_idx:
                    bull_power_falling = curr_bull_power < bull_power[i-1]
                    if curr_bear_power < 0 and bull_power_falling and is_bearish_regime:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Bull Power <= 0 OR Bear Power falling (more negative) OR regime changes to bearish
            bear_power_falling = (i == start_idx) or (curr_bear_power < bear_power[i-1])
            if (curr_bull_power <= 0) or bear_power_falling or (not is_bullish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Bear Power >= 0 OR Bull Power rising (more positive) OR regime changes to bullish
            bull_power_rising = (i == start_idx) or (curr_bull_power > bull_power[i-1])
            if (curr_bear_power >= 0) or bull_power_rising or (not is_bearish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals