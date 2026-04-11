#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w volume confirmation + ADX trend filter
# - Donchian levels from 1d: upper/lower bands act as dynamic support/resistance
# - Long when price breaks above upper band with volume > 1.5x 20-period average (strong conviction)
# - Short when price breaks below lower band with volume > 1.5x 20-period average
# - ADX trend filter: only trade when ADX(14) > 20 to ensure trending conditions and avoid chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 10-30 trades/year (40-120 total over 4 years) to stay within fee drag limits for 1d
# - Volume requirement (>1.5x average) ensures we only trade high-conviction breakouts
# - ADX filter prevents false signals in ranging markets
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1w HTF provides reliable volume confirmation and ADX, reducing false signals from daily noise

name = "1d_1w_donchian_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w volume and ADX
    volume_1w = df_1w['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w volume SMA (20-period)
    volume_series = pd.Series(volume_1w)
    volume_sma_20_1w = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 1w
    # True Range
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = pd.Series(low_1w).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w indicators to 1d timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_upper[i-1]  # Close above previous period's upper band
        breakout_short = price_close < donchian_lower[i-1]  # Close below previous period's lower band
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1w aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ADX trend filter: trade only when ADX > 20 (trending market)
        trend_filter = adx_aligned[i] > 20
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + volume confirmation + trend filter
        if breakout_long and vol_confirm and trend_filter:
            enter_long = True
        
        # Short: Donchian lower breakdown + volume confirmation + trend filter
        if breakout_short and vol_confirm and trend_filter:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or trend weakness
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below lower band OR trend weakens
            exit_long = (price_close < donchian_lower[i-1]) or (not trend_filter)
        elif position == -1:
            # Exit short if price breaks above upper band OR trend weakens
            exit_short = (price_close > donchian_upper[i-1]) or (not trend_filter)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals