#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > weekly EMA34 (uptrend) AND volume > 2.0x 20-bar average.
# Short when Williams %R > -20 (overbought) AND price < weekly EMA34 (downtrend) AND volume > 2.0x 20-bar average.
# Exit when Williams %R crosses -50 (mean reversion midpoint) OR ATR-based stoploss (2.0x ATR).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Williams %R captures short-term extremes; weekly EMA34 filters trend; volume spike confirms conviction.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via weekly EMA34 trend filter.

name = "6h_WilliamsR_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 14, 20, 14)  # warmup for weekly EMA34, Williams %R, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: oversold (Williams %R < -80), uptrend (price > weekly EMA34), volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > ema_34_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: overbought (Williams %R > -20), downtrend (price < weekly EMA34), volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < ema_34_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R crosses -50 (mean reversion) OR ATR stoploss
            exit_signal = False
            if curr_williams_r > -50:  # crossed above -50
                exit_signal = True
            elif curr_close < entry_price - 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R crosses -50 (mean reversion) OR ATR stoploss
            exit_signal = False
            if curr_williams_r < -50:  # crossed below -50
                exit_signal = True
            elif curr_close > entry_price + 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals