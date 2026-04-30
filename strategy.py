#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Uses 12h HTF for Donchian channel calculation (20-period) to identify structural breakouts
# and 1d EMA50 for trend filter to avoid counter-trend trades.
# Long when price breaks above 12h Donchian upper channel in uptrend (6h close > 1d EMA50) with volume spike (>1.8x average).
# Short when price breaks below 12h Donchian lower channel in downtrend (6h close < 1d EMA50) with volume spike.
# Designed for low trade frequency (~12-30/year on 6h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>1.8x average) to balance signal quality and trade count.
# Stoploss at 2.5 * ATR and take profit at 2.0 * ATR to allow for proper risk-reward in 6h timeframe.
# Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
# Focus on BTC/ETH as primary targets with SOL as secondary confirmation.

name = "6h_12hDonchian20_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for Donchian calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper: max(high, lookback=20)
    # Donchian lower: min(low, lookback=20)
    lookback = 20
    donchian_upper = np.full_like(high_12h, np.nan)
    donchian_lower = np.full_like(low_12h, np.nan)
    
    for i in range(lookback, len(high_12h)):
        donchian_upper[i] = np.max(high_12h[i-lookback:i])
        donchian_lower[i] = np.min(low_12h[i-lookback:i])
    
    # Align 12h Donchian levels to 6h timeframe (wait for 12h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for dynamic stoploss/takeprofit on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(60, 50)  # warmup for EMA(50) and Donchian
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 30-period average (moderate to balance frequency)
        if i >= 30:
            vol_ma_30 = np.mean(volume[i-30:i])
        elif i > 0:
            vol_ma_30 = np.mean(volume[:i])
        else:
            vol_ma_30 = 0
        volume_spike = volume[i] > (1.8 * vol_ma_30) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_dc_upper = donchian_upper_aligned[i]
        curr_dc_lower = donchian_lower_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike and not np.isnan(curr_dc_upper) and not np.isnan(curr_dc_lower):
                # Bullish entry: price breaks above 12h Donchian upper with 1d uptrend (close > EMA50)
                if curr_close > curr_dc_upper and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 12h Donchian lower with 1d downtrend (close < EMA50)
                elif curr_close < curr_dc_lower and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0 * ATR above entry
            elif curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0 * ATR below entry
            elif curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals