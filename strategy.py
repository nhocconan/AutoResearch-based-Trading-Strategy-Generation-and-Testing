#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above H3 AND price > 1w EMA50 AND volume > 2.0x 24-bar average.
# Short when price breaks below L3 AND price < 1w EMA50 AND volume > 2.0x 24-bar average.
# Uses discrete sizing 0.25 to balance return and drawdown. No session filter to maximize opportunities.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# 1w EMA50 provides robust trend alignment that works in both bull (price above EMA) and bear (price below EMA).
# Camarilla H3/L3 levels offer reliable breakout points with lower noise than H4/L4.
# Volume confirmation (2.0x average) ensures only high-conviction breakouts are traded.

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 1w trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Calculate Camarilla levels (based on previous day's range)
    # We need to compute daily levels from 1d data, but since we're on 1d timeframe,
    # we can use the previous day's high/low/close directly
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):  # start from 1 to have previous day
        # Previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla H3 = close + (high - low) * 1.1/4
        # Camarilla L3 = close - (high - low) * 1.1/4
        camarilla_h3[i] = prev_close + (prev_high - prev_low) * 1.1 / 4
        camarilla_l3[i] = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume confirmation: current 1d volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_h3[i]  # break above H3
        breakout_down = curr_low < camarilla_l3[i]  # break below L3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above H3 AND price > 1w EMA50 AND volume confirmation
            if (breakout_up and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below L3 AND price < 1w EMA50 AND volume confirmation
            elif (breakout_down and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below L3 (stoploss) OR price < 1w EMA50 (trend change)
            if (curr_low < camarilla_l3[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above H3 (stoploss) OR price > 1w EMA50 (trend change)
            if (curr_high > camarilla_h3[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals