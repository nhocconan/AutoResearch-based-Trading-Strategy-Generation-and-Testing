#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI reversal with weekly trend filter for mean reversion
# Weekly RSI > 50 defines bullish bias (long bias), < 50 bearish bias (short bias)
# Daily RSI < 30 triggers long in bullish bias, > 70 triggers short in bearish bias
# Exit when daily RSI returns to neutral zone (40-60)
# Works in bull markets: buy dips in uptrend (weekly RSI > 50 + daily oversold)
# Works in bear markets: sell rallies in downtrend (weekly RSI < 50 + daily overbought)
# Low frequency: daily signals only, ~10-25 trades/year expected

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE for trend bias
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI (14 periods)
    rsi_len = 14
    delta = np.diff(df_1w['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_len] = np.nanmean(gain[1:rsi_len+1])
    avg_loss[rsi_len] = np.nanmean(loss[1:rsi_len+1])
    
    for i in range(rsi_len+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_len-1) + gain[i]) / rsi_len
        avg_loss[i] = (avg_loss[i-1] * (rsi_len-1) + loss[i]) / rsi_len
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([np.full(rsi_len, np.nan), rsi_1w])
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily RSI (14 periods)
    delta_daily = np.diff(close, prepend=np.nan)
    gain_daily = np.where(delta_daily > 0, delta_daily, 0)
    loss_daily = np.where(delta_daily < 0, -delta_daily, 0)
    
    avg_gain_daily = np.zeros_like(gain_daily)
    avg_loss_daily = np.zeros_like(loss_daily)
    avg_gain_daily[rsi_len] = np.nanmean(gain_daily[1:rsi_len+1])
    avg_loss_daily[rsi_len] = np.nanmean(loss_daily[1:rsi_len+1])
    
    for i in range(rsi_len+1, len(gain_daily)):
        avg_gain_daily[i] = (avg_gain_daily[i-1] * (rsi_len-1) + gain_daily[i]) / rsi_len
        avg_loss_daily[i] = (avg_loss_daily[i-1] * (rsi_len-1) + loss_daily[i]) / rsi_len
    
    rs_daily = np.divide(avg_gain_daily, avg_loss_daily, out=np.full_like(avg_gain_daily, np.nan), where=avg_loss_daily!=0)
    rsi_daily = 100 - (100 / (1 + rs_daily))
    rsi_daily = np.concatenate([np.full(rsi_len, np.nan), rsi_daily])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 2 * rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(rsi_daily[i])):
            signals[i] = 0.0
            continue
        
        weekly_rsi = rsi_1w_aligned[i]
        daily_rsi = rsi_daily[i]
        
        if position == 0:
            # Enter long: weekly bullish bias + daily oversold
            if weekly_rsi > 50 and daily_rsi < 30:
                position = 1
                signals[i] = position_size
            # Enter short: weekly bearish bias + daily overbought
            elif weekly_rsi < 50 and daily_rsi > 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: daily RSI returns to neutral (40-60)
            if 40 <= daily_rsi <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: daily RSI returns to neutral (40-60)
            if 40 <= daily_rsi <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wRSI_1dRSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0