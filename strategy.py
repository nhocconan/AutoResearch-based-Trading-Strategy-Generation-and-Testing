#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14017_4h_donchian20_1d_vol_volatility_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(values, span):
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values

def calculate_bollinger_bands(close, period, std_dev):
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower, sma

def calculate_adx(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    plus_di = 100 * (plus_dm_sum / tr_sum)
    minus_di = 100 * (minus_dm_sum / tr_sum)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volatility regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR and Bollinger Bands for volatility regime
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    bb_upper_1d, bb_lower_1d, bb_middle_1d = calculate_bollinger_bands(df_1d['close'].values, 20, 2.0)
    
    # Calculate volatility regime: low volatility when BB width < 1d ATR
    bb_width_1d = bb_upper_1d - bb_lower_1d
    low_vol_regime = bb_width_1d < atr_1d  # Low volatility regime
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # 4h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(low_vol_regime_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Donchian breakout signals (using previous bar's bands)
        breakout_up = close[i] > donchian_upper[i-1]
        breakout_down = close[i] < donchian_lower[i-1]
        
        # Generate signals only in low volatility regime
        if low_vol_regime_aligned[i]:
            if position == 0:
                if breakout_up and volume_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (2.0 * atr[i])
                elif breakout_down and volume_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (2.0 * atr[i])
                else:
                    signals[i] = 0.0
            elif position == 1:
                if close[i] <= stop_price or breakout_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] >= stop_price or breakout_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In high volatility regime, stay flat
            signals[i] = 0.0
            if position != 0:
                position = 0
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14017_4h_donchian20_1d_vol_volatility_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(values, span):
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values

def calculate_bollinger_bands(close, period, std_dev):
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower, sma

def calculate_adx(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    plus_di = 100 * (plus_dm_sum / tr_sum)
    minus_di = 100 * (minus_dm_sum / tr_sum)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volatility regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR and Bollinger Bands for volatility regime
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    bb_upper_1d, bb_lower_1d, bb_middle_1d = calculate_bollinger_bands(df_1d['close'].values, 20, 2.0)
    
    # Calculate volatility regime: low volatility when BB width < 1d ATR
    bb_width_1d = bb_upper_1d - bb_lower_1d
    low_vol_regime = bb_width_1d < atr_1d  # Low volatility regime
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # 4h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(low_vol_regime_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Donchian breakout signals (using previous bar's bands)
        breakout_up = close[i] > donchian_upper[i-1]
        breakout_down = close[i] < donchian_lower[i-1]
        
        # Generate signals only in low volatility regime
        if low_vol_regime_aligned[i]:
            if position == 0:
                if breakout_up and volume_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (2.0 * atr[i])
                elif breakout_down and volume_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (2.0 * atr[i])
                else:
                    signals[i] = 0.0
            elif position == 1:
                if close[i] <= stop_price or breakout_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] >= stop_price or breakout_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In high volatility regime, stay flat
            signals[i] = 0.0
            if position != 0:
                position = 0
    
    return signals

</think>