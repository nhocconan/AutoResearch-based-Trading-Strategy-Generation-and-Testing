#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ADX regime filter
    # Long when price breaks above Camarilla H3 level + 1d volume > 1.5x 20-period average + 1d ADX > 20
    # Short when price breaks below Camarilla L3 level + 1d volume > 1.5x 20-period average + 1d ADX > 20
    # Exit when price crosses 20-period EMA in opposite direction
    # Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown
    # Target: 75-150 total trades over 4 years (~19-37/year) to avoid fee drag
    # Camarilla provides clear support/resistance levels from prior day's range
    # Volume filter ensures breakouts occur with institutional participation
    # ADX filter ensures we only trade in trending markets, avoiding chop
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) with min_periods
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        for i in range(period, len(high)):
            plus_di[i] = 100 * (plus_dm[i] / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm[i] / atr[i]) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d Camarilla levels (based on prior day's high, low, close)
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.1*(High-Low)
    # H2 = Close + 0.5*(High-Low)
    # H1 = Close + 0.25*(High-Low)
    # L1 = Close - 0.25*(High-Low)
    # L2 = Close - 0.5*(High-Low)
    # L3 = Close - 1.1*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-calculate 4h EMA for exit
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        volume_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        
        # Volume filter: current 1d volume > 1.5 * 20-period average
        volume_confirmation = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # ADX filter: trending market (ADX > 20)
        trending_market = adx_aligned[i] > 20
        
        # Breakout conditions with filters
        bullish_breakout = (close[i] > camarilla_h3_aligned[i] and 
                           volume_confirmation and 
                           trending_market)
        bearish_breakout = (close[i] < camarilla_l3_aligned[i] and 
                           volume_confirmation and 
                           trending_market)
        
        # Exit conditions: cross 20-period EMA in opposite direction
        long_exit = close[i] < ema_20[i]
        short_exit = close[i] > ema_20[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0