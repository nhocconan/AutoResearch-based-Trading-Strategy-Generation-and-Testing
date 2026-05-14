#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3, S3) from daily candles for stronger breakout signals
# 1d EMA34 provides strong trend filter to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms breakout validity
# ATR-based stoploss (1.5x ATR) manages risk with tight stops
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag
# Works in bull markets via R3 breakouts and in bear markets via S3 breakdowns with trend alignment

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 2 previous days for Camarilla calculation (12h bars: 2 days = 4 bars)
        if i < 4:  # 2 * 12h bars = 4 bars for 2 days
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 1d candle (2*12h bars ago)
        prev_day_idx = i - 4
        if prev_day_idx < 0:
            signals[i] = 0.0
            continue
            
        # Previous day's OHLC for Camarilla calculation
        phigh = np.max(high[prev_day_idx:prev_day_idx+2])
        plow = np.min(low[prev_day_idx:prev_day_idx+2])
        pclose = close[prev_day_idx+1]  # close of previous day
        
        # Camarilla R3 and S3 levels (stronger breakout levels)
        rang = phigh - plow
        r3 = pclose + rang * 1.1 / 4
        s3 = pclose - rang * 1.1 / 4
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 1.5 * ATR below entry
            stop_price = entry_price - 1.5 * curr_atr
            # Exit conditions: price below S3 OR price below 1d EMA34 OR stoploss hit
            if curr_close < s3 or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 1.5 * ATR above entry
            stop_price = entry_price + 1.5 * curr_atr
            # Exit conditions: price above R3 OR price above 1d EMA34 OR stoploss hit
            if curr_close > r3 or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1d EMA34 AND volume spike
            if curr_high > r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif curr_low < s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals