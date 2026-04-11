#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_squeeze_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily OHLC for Keltner Channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Keltner Channels: EMA(20) +/- ATR(10)*2
    close_series = pd.Series(close_1d)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10
    
    # Bollinger Bands: SMA(20) +/- 2*STDEV(20)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Squeeze condition: Bollinger Bands inside Keltner Channels
    squeeze = (bb_upper <= keltner_upper) & (bb_lower >= keltner_lower)
    
    # Align all indicators to 4h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Daily volume for confirmation
    volume_sma_10 = pd.Series(df_1d['volume'].values).rolling(window=10, min_periods=10).mean().values
    volume_sma_10_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_10)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(atr_10_aligned[i]) or np.isnan(volume_sma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 10-period daily average
        vol_confirm = volume_current > 1.5 * volume_sma_10_aligned[i]
        
        # Squeeze breakout: price breaks out of Bollinger Bands after squeeze
        # Need previous day's BB values to detect breakout
        if i == 100:
            prev_bb_upper = bb_upper[0] if not np.isnan(bb_upper[0]) else 0
            prev_bb_lower = bb_lower[0] if not np.isnan(bb_lower[0]) else 0
        else:
            prev_bb_upper = bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else 0
            prev_bb_lower = bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else 0
        
        # Breakout conditions
        breakout_up = price_close > bb_upper[i-1] if i > 0 and not np.isnan(bb_upper[i-1]) else False
        breakout_down = price_close < bb_lower[i-1] if i > 0 and not np.isnan(bb_lower[i-1]) else False
        
        # Only trade breakouts that occur after a squeeze
        squeeze_active = squeeze_aligned[i] if not np.isnan(squeeze_aligned[i]) else False
        squeeze_recent = False
        # Check if squeeze was active in the last 3 days
        for j in range(1, 4):
            if i - j >= 0 and not np.isnan(squeeze_aligned[i-j]) and squeeze_aligned[i-j]:
                squeeze_recent = True
                break
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Breakout above BB upper after squeeze + volume + above EMA20
        if breakout_up and squeeze_recent and vol_confirm and price_close > ema_20_aligned[i]:
            enter_long = True
        
        # Short: Breakout below BB lower after squeeze + volume + below EMA20
        if breakout_down and squeeze_recent and vol_confirm and price_close < ema_20_aligned[i]:
            enter_short = True
        
        # Exit conditions: price returns to EMA20 or opposite Bollinger Band
        exit_long = price_close < ema_20_aligned[i] or price_close < bb_lower[i]
        exit_short = price_close > ema_20_aligned[i] or price_close > bb_upper[i]
        
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

# Hypothesis: Keltner Squeeze breakout on daily timeframe with volume confirmation and EMA20 trend filter.
# The squeeze occurs when Bollinger Bands contract inside Keltner Channels, indicating low volatility.
# Breakouts from the squeeze often lead to strong directional moves. Works in both bull and bear markets
# by capturing volatility expansion after contraction. Volume confirmation ensures participation.
# EMA20 filter avoids counter-trend trades. Position size 0.25 balances risk and return.
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves.