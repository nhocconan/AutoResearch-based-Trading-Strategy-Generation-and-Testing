#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout + 1d EMA34 trend filter + volume spike confirmation.
# Long when price breaks above Camarilla R4 AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S4 AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Exit when price crosses Camarilla H4/L4 midline OR ATR-based stoploss (2.5x ATR).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Camarilla R4/S4 levels provide stronger intraday support/resistance than R3/S3, reducing false breakouts.
# 1d EMA34 filters trend effectively across bull/bear markets.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via 1d EMA34 trend filter.

name = "4h_Camarilla_R4S4_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data for pivot)
    # Use previous day's OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: based on previous day's range
    rang = prev_high - prev_low
    camarilla_h4 = prev_close + rang * 1.1 / 2
    camarilla_h3 = prev_close + rang * 1.1 / 4
    camarilla_l3 = prev_close - rang * 1.1 / 4
    camarilla_l4 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_l4_mid = (camarilla_h4_aligned + camarilla_l4_aligned) / 2
    
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
    
    start_idx = max(50, 20, 14)  # warmup for EMA34, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h4_l4_mid[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla H4, uptrend (price > 1d EMA34), volume confirmation
            if (curr_high > camarilla_h4_aligned[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Camarilla S4, downtrend (price < 1d EMA34), volume confirmation
            elif (curr_low < camarilla_l4_aligned[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Camarilla H4/L4 midline cross OR ATR stoploss
            exit_signal = False
            if curr_close < camarilla_h4_l4_mid[i]:  # midline cross
                exit_signal = True
            elif curr_close < entry_price - 2.5 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Camarilla H4/L4 midline cross OR ATR stoploss
            exit_signal = False
            if curr_close > camarilla_h4_l4_mid[i]:  # midline cross
                exit_signal = True
            elif curr_close > entry_price + 2.5 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals