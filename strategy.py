#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - Donchian levels from 4h: upper/lower bands act as dynamic support/resistance
# - Long when price breaks above upper band with 1d volume > 1.5x 20-period average AND 1w close > 1w SMA(50) (bullish weekly trend)
# - Short when price breaks below lower band with 1d volume > 1.5x 20-period average AND 1w close < 1w SMA(50) (bearish weekly trend)
# - ATR stoploss: exit when price moves 2.5*ATR against position from entry
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Weekly trend filter ensures we only trade in direction of higher timeframe momentum, reducing false breakouts
# - Volume confirmation ensures breakouts have participation, avoiding low-conviction moves

name = "4h_1d_1w_donchian_volume_trend_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for ATR-based stoploss
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1w close and SMA(50) for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    sma_50_1w = close_1w_series.rolling(window=50, min_periods=50).mean().values
    
    # Align 1d and 1w indicators to 4h timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(close_1w_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(atr_14_4h[i])):
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
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Weekly trend filter: 1w close above/below 50-period SMA
        weekly_uptrend = close_1w_aligned[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < sma_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + volume confirmation + weekly uptrend
        if breakout_long and vol_confirm and weekly_uptrend:
            enter_long = True
        
        # Short: Donchian lower breakdown + volume confirmation + weekly downtrend
        if breakout_short and vol_confirm and weekly_downtrend:
            enter_short = True
        
        # Exit conditions: ATR-based stoploss or opposite Donchian breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops 2.5*ATR below entry price OR breaks below lower band
            exit_long = (price_close < entry_price - 2.5 * atr_14_4h[i]) or (price_close < donchian_lower[i-1])
        elif position == -1:
            # Exit short if price rises 2.5*ATR above entry price OR breaks above upper band
            exit_short = (price_close > entry_price + 2.5 * atr_14_4h[i]) or (price_close > donchian_upper[i-1])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals