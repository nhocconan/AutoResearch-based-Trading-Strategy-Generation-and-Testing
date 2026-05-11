# 1d_1w_RSI_Divergence_Trend
# Hypothesis: Weekly RSI divergence combined with daily price action for high-probability reversals.
# Long when: weekly RSI makes higher low while price makes lower low (bullish divergence) AND daily close > daily EMA50.
# Short when: weekly RSI makes lower high while price makes higher high (bearish divergence) AND daily close < daily EMA50.
# Exit when divergence fails or price crosses opposite EMA50.
# Works in bull by catching pullbacks in uptrend; works in bear by selling rallies in downtrend.
# Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag.

name = "1d_1w_RSI_Divergence_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for RSI divergence
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI(14) ---
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 14:
            if i == 0:
                avg_gain[i] = np.nan
                avg_loss[i] = np.nan
            else:
                avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else np.nan
                avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # --- Daily EMA50 ---
    ema_50 = np.full(n, np.nan)
    for i in range(n):
        if i < 50:
            ema_50[i] = np.nan
        elif i == 50:
            ema_50[i] = np.mean(close[0:50])
        else:
            ema_50[i] = (close[i] * 2 / (50 + 1)) + (ema_50[i-1] * (49 / (50 + 1)))
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track weekly pivot points for divergence detection
    # We'll look for swing points in weekly data
    start_idx = 50  # Need EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find corresponding weekly index for current daily bar
        # Since we aligned, we can check for divergence conditions
        bullish_divergence = False
        bearish_divergence = False
        
        # Check for bullish divergence: price makes lower low, RSI makes higher low
        if i >= 7:  # Need at least a week of data
            # Look back 1-4 weeks for swing points
            lookback_weeks = min(4, i // 7)
            for weeks_back in range(1, lookback_weeks + 1):
                curr_week_start = i - (weeks_back * 7)
                prev_week_start = curr_week_start - 7
                
                if prev_week_start < 0:
                    break
                
                # Find weekly lows
                week_low_1 = np.min(low[max(0, curr_week_start-6):curr_week_start+1])
                week_low_2 = np.min(low[max(0, prev_week_start-6):prev_week_start+1])
                
                # Get corresponding weekly RSI values
                rsi_idx_1 = curr_week_start
                rsi_idx_2 = prev_week_start
                
                if (rsi_idx_1 < len(rsi_1w_aligned) and rsi_idx_2 < len(rsi_1w_aligned) and
                    not np.isnan(rsi_1w_aligned[rsi_idx_1]) and not np.isnan(rsi_1w_aligned[rsi_idx_2])):
                    # Bullish divergence: price lower low, RSI higher low
                    if week_low_1 < week_low_2 and rsi_1w_aligned[rsi_idx_1] > rsi_1w_aligned[rsi_idx_2]:
                        bullish_divergence = True
                        break
                    # Bearish divergence: price higher high, RSI lower high
                    week_high_1 = np.max(high[max(0, curr_week_start-6):curr_week_start+1])
                    week_high_2 = np.max(high[max(0, prev_week_start-6):prev_week_start+1])
                    if week_high_1 > week_high_2 and rsi_1w_aligned[rsi_idx_1] < rsi_1w_aligned[rsi_idx_2]:
                        bearish_divergence = True
                        break
        
        if position == 0:
            if bullish_divergence and close[i] > ema_50[i]:
                # Long: bullish divergence with price above EMA50
                signals[i] = 0.25
                position = 1
            elif bearish_divergence and close[i] < ema_50[i]:
                # Short: bearish divergence with price below EMA50
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: bearish divergence or price crosses below EMA50
                if bearish_divergence or close[i] < ema_50[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish divergence or price crosses above EMA50
                if bullish_divergence or close[i] > ema_50[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals