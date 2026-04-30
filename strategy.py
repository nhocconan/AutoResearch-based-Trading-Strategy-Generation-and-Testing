#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Bollinger Band squeeze breakout with volume confirmation and 1d trend filter
# Bollinger Band squeeze indicates low volatility and potential for explosive moves.
# Breakout above upper band or below lower band with volume spike captures the move.
# 1d EMA(50) ensures alignment with longer-term trend to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
# Uses 4h timeframe as requested, with 1d HTF for trend filter.

name = "4h_BollingerSqueeze_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std * bb_std_dev)
    bb_lower = bb_ma - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: bb_width < 20-period average of bb_width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(bb_period, 20) + 14  # warmup for BB and ATR
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_squeeze = bb_squeeze[i]
        
        if position == 0:  # Flat - look for new entries
            # Require Bollinger Band squeeze and volume spike
            if curr_bb_squeeze and volume_spike:
                # Bullish entry: price breaks above upper Bollinger Band with 1d uptrend
                if curr_close > curr_bb_upper and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below lower Bollinger Band with 1d downtrend
                elif curr_close < curr_bb_lower and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks below lower Bollinger Band
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_bb_lower:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches upper Bollinger Band
            elif curr_close >= curr_bb_upper:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks above upper Bollinger Band
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_bb_upper:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches lower Bollinger Band
            elif curr_close <= curr_bb_lower:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals