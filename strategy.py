#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise support/resistance based on prior day's range
# Breakout above R1 or below S1 with volume confirmation indicates strong momentum
# 1d EMA34 > EMA34 ensures alignment with longer-term trend to avoid whipsaw in bear markets
# Volume spike (2.0x 24-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter.
# Added ATR-based stoploss to manage risk during adverse moves.

name = "4h_Camarilla_R1S1_1dEMA34_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels based on prior day's OHLC
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.125 / 4)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.125 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*4h = 96h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR for stoploss (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 24, 14)  # warmup for EMA, volume MA, and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_24[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and price above/below EMA34 for trend alignment
            if curr_volume_spike:
                # Bullish entry: break above R1 with price > EMA34_1d
                if curr_close > curr_r1 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S1 with price < EMA34_1d
                elif curr_close < curr_s1 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2x ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit when price drops below R1 (breakout fails), price crosses below EMA34_1d, or stoploss hit
            if curr_close < curr_r1 or curr_close < curr_ema_34_1d or curr_close < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2x ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit when price rises above S1 (breakdown fails), price crosses above EMA34_1d, or stoploss hit
            if curr_close > curr_s1 or curr_close > curr_ema_34_1d or curr_close > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals