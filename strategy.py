#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # need sufficient data for weekly indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend - needs to confirm on weekly close
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Weekly RSI(14) for momentum confirmation
    delta = pd.Series(close_weekly).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_weekly = 100 - (100 / (1 + rs))
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly.values)
    
    # Daily ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(rsi_weekly_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + RSI not overbought + price pullback to EMA + volume
            if (close[i] > ema_weekly_aligned[i] and 
                rsi_weekly_aligned[i] < 70 and 
                close[i] <= ema_weekly_aligned[i] + 0.5 * atr[i] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + RSI not oversold + price bounce to EMA + volume
            elif (close[i] < ema_weekly_aligned[i] and 
                  rsi_weekly_aligned[i] > 30 and 
                  close[i] >= ema_weekly_aligned[i] - 0.5 * atr[i] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Weekly trend reversal or RSI extreme
            if position == 1:
                # Exit long: weekly downtrend or RSI overbought
                if (close[i] < ema_weekly_aligned[i] or 
                    rsi_weekly_aligned[i] >= 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: weekly uptrend or RSI oversold
                if (close[i] > ema_weekly_aligned[i] or 
                    rsi_weekly_aligned[i] <= 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyEMA_RSI_Pullback_Volume"
timeframe = "1d"
leverage = 1.0