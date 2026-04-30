#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams Fractal breakout with volume confirmation and 12h EMA(50) trend filter
# Williams Fractals identify significant swing highs/lows that act as support/resistance.
# Breakouts above recent bullish fractal or below bearish fractal with volume spike indicate strong momentum.
# 12h EMA(50) filters trades to align with higher-timeframe trend, reducing false breakouts.
# Designed for low trade frequency (~20-50/year on 4h) to minimize fee drag and improve bear market performance.

name = "4h_WilliamsFractal_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Williams Fractal calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h Williams Fractals (requires 5-bar window: 2 bars left/right)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    bullish_fractal = np.full(len(high_12h), np.nan)
    bearish_fractal = np.full(len(high_12h), np.nan)
    
    # Williams Fractal: bullish = low with two higher lows on each side
    # bearish = high with two lower highs on each side
    for i in range(2, len(high_12h) - 2):
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i-2] and
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i-2] and
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
    
    # Forward fill fractal levels to use until next fractal forms
    bullish_fractal = pd.Series(bullish_fractal).ffill().values
    bearish_fractal = pd.Series(bearish_fractal).ffill().values
    
    # Align Williams Fractal levels to 4h timeframe with 2-bar extra delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h_s = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 52  # warmup for EMA(50) + fractal calculation
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_bullish = bullish_fractal_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike and not (np.isnan(curr_bullish) or np.isnan(curr_bearish)):
                # Bullish entry: price breaks above bullish fractal with 12h uptrend
                if curr_close > curr_bullish and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below bearish fractal with 12h downtrend
                elif curr_close < curr_bearish and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks bearish fractal
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_bearish:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches bullish fractal (mean reversion tendency)
            elif curr_close >= curr_bullish:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks bullish fractal
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_bullish:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches bearish fractal (mean reversion tendency)
            elif curr_close <= curr_bearish:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals