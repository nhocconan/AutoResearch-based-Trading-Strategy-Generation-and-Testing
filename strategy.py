#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses 10-day Donchian midline OR ATR-based stoploss (2.5x ATR).
# Uses discrete position sizing (0.30) to limit drawdown and fee churn.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull/bear via 1w EMA50 trend filter.

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Calculate 20-period Donchian channels (using data up to i-1 to avoid look-ahead)
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                donchian_low = np.min(low[i-20:i])
                donchian_mid = (donchian_high + donchian_low) / 2.0
                
                # Long: break above 20-day high, uptrend (price > 1w EMA50), volume confirmation
                if (curr_high > donchian_high and 
                    curr_close > ema_50_1w_aligned[i] and 
                    curr_volume_confirm):
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Short: break below 20-day low, downtrend (price < 1w EMA50), volume confirmation
                elif (curr_low < donchian_low and 
                      curr_close < ema_50_1w_aligned[i] and 
                      curr_volume_confirm):
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                else:
                    # Hold flat
                    signals[i] = 0.0
            else:
                # Not enough data for Donchian calculation
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Calculate 10-period Donchian midline for exit
            if i >= 10:
                donchian_high_10 = np.max(high[i-10:i])
                donchian_low_10 = np.min(low[i-10:i])
                donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
            else:
                donchian_mid_10 = curr_close  # fallback to avoid false exit
            
            # Exit conditions: 10-day Donchian midline cross OR ATR stoploss
            exit_signal = False
            if curr_close < donchian_mid_10:  # midline cross
                exit_signal = True
            elif curr_close < entry_price - 2.5 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Calculate 10-period Donchian midline for exit
            if i >= 10:
                donchian_high_10 = np.max(high[i-10:i])
                donchian_low_10 = np.min(low[i-10:i])
                donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
            else:
                donchian_mid_10 = curr_close  # fallback to avoid false exit
            
            # Exit conditions: 10-day Donchian midline cross OR ATR stoploss
            exit_signal = False
            if curr_close > donchian_mid_10:  # midline cross
                exit_signal = True
            elif curr_close > entry_price + 2.5 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals