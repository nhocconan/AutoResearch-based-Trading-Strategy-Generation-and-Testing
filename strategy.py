#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure bull/bear strength.
# 1d ADX regime filter: ADX > 25 = trending (trade Elder Ray extremes), ADX < 20 = range (fade Elder Ray extremes).
# Discrete sizing 0.25 to balance profit and fee drag. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear via regime adaptation: trend following in trending markets, mean reversion in ranging markets.

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # Calculate 1d ADX(14) for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h ATR(14) for stoploss
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 13, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_14_6h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        curr_atr = atr_14_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Regime-based logic
            if curr_adx > 25:  # Trending regime - trend follow
                # Strong bull power = long
                if curr_bull > 0 and curr_bull > np.nanpercentile(bull_power[max(0,i-50):i], 70):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Strong bear power = short
                elif curr_bear < 0 and curr_bear < np.nanpercentile(bear_power[max(0,i-50):i], 30):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            elif curr_adx < 20:  # Range regime - mean revert
                # Fade extreme bull power (overbought)
                if curr_bull > np.nanpercentile(bull_power[max(0,i-50):i], 85):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                # Fade extreme bear power (oversold)
                elif curr_bear < np.nanpercentile(bear_power[max(0,i-50):i], 15):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit conditions
            if curr_low <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif curr_adx > 25:  # Trending: exit on bear power divergence
                if curr_bear < 0 and curr_bear < np.nanpercentile(bear_power[max(0,i-20):i], 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Range: exit on mean reversion
                if curr_bull < np.nanpercentile(bull_power[max(0,i-20):i], 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit conditions
            if curr_high >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif curr_adx > 25:  # Trending: exit on bull power divergence
                if curr_bull > 0 and curr_bull > np.nanpercentile(bull_power[max(0,i-20):i], 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Range: exit on mean reversion
                if curr_bear > np.nanpercentile(bear_power[max(0,i-20):i], 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals