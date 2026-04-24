#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 AND 1w EMA34 > 1w EMA89 (bullish regime) AND volume > 1.5x 20-period average volume
- Short when price breaks below Camarilla L3 AND 1w EMA34 < 1w EMA89 (bearish regime) AND volume > 1.5x 20-period average volume
- Exit on opposite Camarilla breakout or when volume drops below average
- Position size fixed at 0.25 to limit fee churn and manage drawdown
- Uses 1d primary with 1w HTF to target 30-100 trades over 4 years (7-25/year)
- Camarilla levels provide institutional support/resistance; EMA34/EMA89 confirms multi-week trend; volume filters breakout validity
- Fixed sizing reduces churn while maintaining exposure during trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla levels (based on previous day's range)
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # We need previous day's OHLC, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    H3 = prev_close + 1.1 * camarilla_range / 4
    L3 = prev_close - 1.1 * camarilla_range / 4
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 90:
        return np.zeros(n)
    
    # Calculate 1w EMA34 and EMA89
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = pd.Series(df_1w['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w indicators to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Trend filter: bullish if EMA34 > EMA89, bearish if EMA34 < EMA89
    bullish_trend = ema_34_1w_aligned > ema_89_1w_aligned
    bearish_trend = ema_34_1w_aligned < ema_89_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 89) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_89_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 AND bullish trend AND volume confirmation
            if close[i] > H3[i] and bullish_trend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 AND bearish trend AND volume confirmation
            elif close[i] < L3[i] and bearish_trend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below L3 OR volume drops below average
            if close[i] < L3[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above H3 OR volume drops below average
            if close[i] > H3[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_1wEMA34_EMA89_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0