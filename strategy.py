#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 4h close > EMA50 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND 4h close < EMA50 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.20 to manage drawdown and reduce trade frequency.
# Primary timeframe: 1h, HTF: 4h for EMA trend and Camarilla calculation (from previous 4h bar).
# Session filter: 08-20 UTC to avoid low-liquidity periods.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) - prices.index is already DatetimeIndex
    session_hours = prices.index.hour
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar's OHLC
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # where C = (H+L+O)/3 (typical price)
    prev_close_4h = df_4h['close'].values
    prev_high_4h = df_4h['high'].values
    prev_low_4h = df_4h['low'].values
    prev_open_4h = df_4h['open'].values
    
    # Typical price (pivot) for previous 4h bar
    pivot_4h = (prev_high_4h + prev_low_4h + prev_open_4h) / 3.0
    camarilla_r3 = pivot_4h + ((prev_high_4h - prev_low_4h) * 1.1 / 4)
    camarilla_s3 = pivot_4h - ((prev_high_4h - prev_low_4h) * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (waits for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # 4h EMA50 trend filter
    ema_50 = pd.Series(prev_close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: current 1h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND bullish trend AND volume confirmation AND session
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND bearish trend AND volume confirmation AND session
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR trend turns bearish OR outside session
            if (curr_low < camarilla_s3_aligned[i] or 
                bearish_trend or 
                not in_session[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR trend turns bullish OR outside session
            if (curr_high > camarilla_r3_aligned[i] or 
                bullish_trend or 
                not in_session[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals