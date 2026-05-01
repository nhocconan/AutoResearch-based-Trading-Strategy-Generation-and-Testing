#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume spike confirmation.
# Long when RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume > 1.5x 20-bar average.
# Short when RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.20 to minimize fee churn. Designed for 1h timeframe to capture mean reversion in ranging markets while respecting higher timeframe trend.
# 4h EMA50 ensures we only trade in the direction of the intermediate trend, reducing false signals in strong trends.
# Volume spike requirement confirms momentum behind the move, filtering low-conviction entries.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.

name = "1h_RSI14_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 calculation
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for RSI and EMA calculation
    
    for i in range(start_idx, n):
        # Session filter: trade only during 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # RSI conditions
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        # Trend filter: price vs 4h EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: RSI oversold AND price above 4h EMA50 AND volume confirmation
            if (rsi_oversold and 
                price_above_ema and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND price below 4h EMA50 AND volume confirmation
            elif (rsi_overbought and 
                  price_below_ema and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete) OR price breaks below 4h EMA50 (trend change)
            if (rsi_values[i] > 50 or 
                curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete) OR price breaks above 4h EMA50 (trend change)
            if (rsi_values[i] < 50 or 
                curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals