#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation
# - Long: Price breaks above Donchian upper channel + 1d ATR > 1d ATR SMA(20) (high volatility regime) + volume > 2.0x 20-period average
# - Short: Price breaks below Donchian lower channel + same volatility + volume conditions
# - Exit: ATR trailing stop (2.5 ATR from extreme) or opposite Donchian breakout
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Volatility regime filter ensures we only trade during strong momentum periods, reducing whipsaw
# - Higher volume threshold (2.0x) increases signal quality and reduces trade frequency
# - ATR trailing stop adapts to volatility and locks in profits during trends

name = "4h_1d_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ATR and its SMA for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_sma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_sma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20_1d)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr_sma_20_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Regime filters: 1d ATR above its 20-period average (high volatility regime)
        vol_regime = atr_1d[i] > atr_sma_20_aligned[i] if not np.isnan(atr_1d[i]) else False
        
        # Volume confirmation: current volume > 2.0x 20-period average (1d aligned)
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above upper Donchian channel with volatility regime and volume confirmation
        if close_price > upper_channel and vol_regime and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below lower Donchian channel with volatility regime and volume confirmation
        if close_price < lower_channel and vol_regime and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below lower channel
            exit_long = (close_price <= long_stop) or (close_price < lower_channel)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above upper channel
            exit_short = (close_price >= short_stop) or (close_price > upper_channel)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.5 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.5 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2.5*ATR)
            long_stop = max(long_stop, high_price - 2.5 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2.5*ATR)
            short_stop = min(short_stop, low_price + 2.5 * atr_14[i])
        
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