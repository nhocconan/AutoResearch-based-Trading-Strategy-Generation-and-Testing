#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with daily volume confirmation.
# Uses daily Camarilla levels (H3/L3) for mean reversion entries in ranging markets,
# filtered by daily ADX trend strength to avoid trending conditions.
# Volume spike confirms institutional participation at pivot levels.
# Works in both bull and bear markets by fading extremes during consolidation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13548_12h_camarilla1d_vol_adx_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * np.where(atr != 0, 
                            np.convolve(plus_dm, np.ones(period)/period, mode='full')[:len(high)] / atr, 
                            0)
    minus_di = 100 * np.where(atr != 0, 
                             np.convolve(minus_dm, np.ones(period)/period, mode='full')[:len(high)] / atr, 
                             0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = np.convolve(dx, np.ones(period)/period, mode='full')[:len(high)]
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3/L3 = close ± 1.1 * (high - low) / 2
    camarilla_h3 = close_1d + (CAMARILLA_MULTIPLIER * (high_1d - low_1d) / 2)
    camarilla_l3 = close_1d - (CAMARILLA_MULTIPLIER * (high_1d - low_1d) / 2)
    
    # Calculate daily ADX for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    
    # Align to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Range condition: ADX below threshold (non-trending)
        is_ranging = adx_1d_aligned[i] < ADX_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Reversal signals at Camarilla H3/L3 levels
        sell_signal = (is_ranging and volume_ok and 
                      close[i] >= camarilla_h3_aligned[i])
        buy_signal = (is_ranging and volume_ok and 
                     close[i] <= camarilla_l3_aligned[i])
        
        # Generate signals
        if position == 0:
            if sell_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            elif buy_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price reaches opposite H3 level or ADX increases
            if close[i] >= camarilla_h3_aligned[i] or adx_1d_aligned[i] >= ADX_THRESHOLD:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price reaches opposite L3 level or ADX increases
            if close[i] <= camarilla_l3_aligned[i] or adx_1d_aligned[i] >= ADX_THRESHOLD:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with weekly EMA trend filter and volume confirmation.
# Uses 12h price channel breakouts aligned with weekly momentum to capture strong trending moves.
# Volume confirmation ensures institutional participation. Works in bull markets (breakouts above upper band)
# and bear markets (breakdowns below lower band). Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13548_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals using Donchian channels
        breakout_up = volume_ok and uptrend and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < lowest_low[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals