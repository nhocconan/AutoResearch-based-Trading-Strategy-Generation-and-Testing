#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
# - Primary: 4h timeframe for optimal trade frequency (19-50/year target)
# - HTF: 1d EMA200 for long-term trend direction (avoid counter-trend trades)
# - Long: Price breaks above Donchian(20) upper band + price > 1d EMA200 + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) lower band + price < 1d EMA200 + volume > 1.5x 20-period MA
# - Exit: Price reverts to Donchian(20) midpoint (mean reversion) or ATR-based stoploss
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-180 total trades over 4 years (25-45/year) - within 4h sweet spot
# - Donchian breakouts work in both trending and ranging markets
# - 1d EMA200 filter ensures we trade with long-term trend, reducing whipsaws in bear markets
# - Volume confirmation increases breakout reliability

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need enough for EMA200
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 4h Donchian Channel (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    # Middle band = (upper + lower) / 2
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h volume moving average (20-period) for volume confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h ATR(14) for dynamic stoploss
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h) - pd.Series(close_4h).shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(200, n):  # Start after warmup period (EMA200 needs 200 bars)
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma_20_4h[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_breakout = close_4h[i] > donchian_upper[i]
        short_breakout = close_4h[i] < donchian_lower[i]
        price_above_ema = close_4h[i] > ema_200_1d_aligned[i]
        price_below_ema = close_4h[i] < ema_200_1d_aligned[i]
        volume_spike = volume_4h[i] > 1.5 * volume_ma_20_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian upper breakout + price above 1d EMA200 + volume spike
            if long_breakout and price_above_ema and volume_spike:
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short entry: Donchian lower breakout + price below 1d EMA200 + volume spike
            elif short_breakout and price_below_ema and volume_spike:
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. ATR-based stoploss (2.5 * ATR)
            # 3. Opposite Donchian breakout (strong reversal signal)
            
            if position == 1:  # Long position
                # Mean reversion exit: price returns to midpoint
                mean_reversion_exit = close_4h[i] < donchian_middle[i]
                # ATR stoploss: price drops below entry - 2.5 * ATR
                stoploss_exit = close_4h[i] < entry_price - 2.5 * atr_4h[i]
                # Opposite breakout: price breaks below Donchian lower
                opposite_breakout = close_4h[i] < donchian_lower[i]
                
                if mean_reversion_exit or stoploss_exit or opposite_breakout:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Mean reversion exit: price returns to midpoint
                mean_reversion_exit = close_4h[i] > donchian_middle[i]
                # ATR stoploss: price rises above entry + 2.5 * ATR
                stoploss_exit = close_4h[i] > entry_price + 2.5 * atr_4h[i]
                # Opposite breakout: price breaks above Donchian upper
                opposite_breakout = close_4h[i] > donchian_upper[i]
                
                if mean_reversion_exit or stoploss_exit or opposite_breakout:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals