#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from oversold, price > 1d EMA34, and volume > 2.0x 20-bar average.
# Short when Williams %R crosses below -20 from overbought, price < 1d EMA34, and volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to avoid overtrading.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in both bull and bear markets when combined with trend filter.

name = "4h_WilliamsR_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 14-period lookback
    highest_high_14 = np.maximum.accumulate(high)
    lowest_low_14 = np.minimum.accumulate(low)
    # For Williams %R, we need the highest high and lowest low over the last 14 periods including current
    # We'll compute it manually in the loop to avoid look-ahead
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R calculation and EMA34
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Calculate Williams %R for the last 14 periods (including current bar)
        if i < 13 + start_idx:  # Need at least 14 bars of data
            signals[i] = 0.0
            continue
            
        # Highest high and lowest low over the last 14 bars (including current)
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        
        # Avoid division by zero
        if highest_high == lowest_low:
            williams_r = -50.0  # Neutral value when range is zero
        else:
            williams_r = -100 * (highest_high - curr_close) / (highest_high - lowest_low)
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (from below) AND price > 1d EMA34 AND volume confirmation
            if (williams_r > -80 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                # Check if we were below -80 in the previous bar to confirm crossover
                if i > start_idx:
                    prev_highest_high = np.max(high[i-14:i])
                    prev_lowest_low = np.min(low[i-14:i])
                    if prev_highest_high == prev_lowest_low:
                        prev_williams_r = -50.0
                    else:
                        prev_williams_r = -100 * (prev_highest_high - close[i-1]) / (prev_highest_high - prev_lowest_low)
                    if prev_williams_r <= -80:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            # Short: Williams %R crosses below -20 (from above) AND price < 1d EMA34 AND volume confirmation
            elif (williams_r < -20 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                # Check if we were above -20 in the previous bar to confirm crossover
                if i > start_idx:
                    prev_highest_high = np.max(high[i-14:i])
                    prev_lowest_low = np.min(low[i-14:i])
                    if prev_highest_high == prev_lowest_low:
                        prev_williams_r = -50.0
                    else:
                        prev_williams_r = -100 * (prev_highest_high - close[i-1]) / (prev_highest_high - prev_lowest_low)
                    if prev_williams_r >= -20:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR price < 1d EMA34 (trend violation)
            if (williams_r < -50 or 
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR price > 1d EMA34 (trend violation)
            if (williams_r > -50 or 
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals