#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R1 AND 4h close > EMA50 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S1 AND 4h close < EMA50 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.20 to manage drawdown and reduce fee churn.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Session filter (08-20 UTC) to avoid low-liquidity hours.
# Camarilla levels provide precise intraday support/resistance, EMA50 filters trend alignment, volume spike confirms momentum.
# Primary timeframe: 1h, HTF: 4h for EMA trend and Camarilla calculation.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar's OHLC
    # Camarilla: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    # where C = (H+L+O)/3 (typical price), but using close as pivot for simplicity
    prev_close = df_4h['close'].values
    prev_high = df_4h['high'].values
    prev_low = df_4h['low'].values
    
    # Camarilla levels based on previous 4h bar
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 trend filter
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: current 1h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
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
        breakout_up = curr_high > camarilla_r1_aligned[i]  # break above R1
        breakout_down = curr_low < camarilla_s1_aligned[i]  # break below S1
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R1 AND bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S1 AND bearish trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S1 (stoploss) OR trend turns bearish
            if (curr_low < camarilla_s1_aligned[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (stoploss) OR trend turns bullish
            if (curr_high > camarilla_r1_aligned[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals