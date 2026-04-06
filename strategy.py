#!/usr/bin/env python3
"""
6h Donchian breakout with 1d trend filter and volume confirmation.
Hypothesis: Breakouts on 6h aligned with daily trend (price > EMA50) and volume concentration
capture medium-term trends while avoiding false breakouts. Volume concentration (>2x average)
confirms institutional participation. Works in bull (breakouts) and bear (breakdowns) with
proper filtering. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14271_6h_donchian20_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume concentration: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_concentration = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR, 50 for EMA)
    start = max(20, 20, 14, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with 1d EMA filter and volume concentration
        # Long: break above upper band + price > 1d EMA + volume concentration
        # Short: break below lower band + price < 1d EMA + volume concentration
        breakout_long = (close[i] > highest_high[i-1]) and (close[i] > ema_1d_aligned[i]) and vol_concentration[i]
        breakout_short = (close[i] < lowest_low[i-1]) and (close[i] < ema_1d_aligned[i]) and vol_concentration[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakdown of lower band
            if close[i] <= stop_price or close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short on stop or breakout of upper band
            if close[i] >= stop_price or close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

</think>

#!/usr/bin/env python3
"""
6h Camarilla pivot strategy with volume confirmation and volatility filter.
Hypothesis: Price reacts at Camarilla levels (H3/L3 for reversal, H4/L4 for breakout)
with volume confirmation works across market regimes. Uses 1d OHLC to calculate levels.
Volume > 1.5x average confirms institutional interest. Volatility filter (ATR ratio)
avoids choppy markets. Target: 80-160 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14271_6h_camarilla_vol_volfilt_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.5 * range_1d
    L4 = close_1d - 1.5 * range_1d
    H3 = close_1d + 1.125 * range_1d
    L3 = close_1d - 1.125 * range_1d
    H2 = close_1d + 0.75 * range_1d
    L2 = close_1d - 0.75 * range_1d
    H1 = close_1d + 0.5 * range_1d
    L1 = close_1d - 0.5 * range_1d
    
    # Align all levels to 6h timeframe (using previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Volatility filter: avoid choppy markets (ATR ratio < 0.8)
    atr = calculate_atr(high, low, close, 14)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    volatility_filter = atr_ratio < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (max of 20 for volume/ATR, 0 for levels)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(H3_aligned[i]) or \
           np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_ratio[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check for exit conditions (reverse at opposite H3/L3)
        if position == 1:  # long position
            # Exit long if price reaches H3 (take profit) or breaks below L3 (stop)
            if close[i] >= H3_aligned[i] or close[i] <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit short if price reaches L3 (take profit) or breaks above H3 (stop)
            if close[i] <= L3_aligned[i] or close[i] >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entry signals with volume and volatility filters
            if volume_filter[i] and volatility_filter[i]:
                # Long near L3/H3 reversal or H4 breakout
                if (abs(close[i] - L3_aligned[i]) < 0.001 * close[i] or  # at L3
                    abs(close[i] - H3_aligned[i]) < 0.001 * close[i] or  # at H3
                    close[i] > H4_aligned[i]):  # breakout above H4
                    signals[i] = 0.25
                    position = 1
                # Short near H3/L3 reversal or L4 breakdown
                elif (abs(close[i] - H3_aligned[i]) < 0.001 * close[i] or  # at H3
                      abs(close[i] - L3_aligned[i]) < 0.001 * close[i] or  # at L3
                      close[i] < L4_aligned[i]):  # breakdown below L4
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
6h Ichimoku Cloud strategy with 1d trend filter and volume confirmation.
Hypothesis: Tenkan/Kijun cross above/below cloud with 1d trend alignment (price vs 1d EMA50)
and volume confirmation captures strong trends while avoiding false signals. Cloud acts as
dynamic support/resistance. Target: 70-140 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14271_6h_ichimoku_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data for Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    tenkan_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    kijun_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    senkou_b_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    senkou_b_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    
    # Chikou Span (Lagging Span): close shifted 22 periods back
    # We'll use current close vs Senkou lines for cloud calculation
    
    # The cloud is between Senkou A and Senkou B
    # For simplicity, we'll use the current values (not shifted) as approximation
    # In practice, Senkou spans are shifted, but for signal generation we compare
    # current price to current cloud boundaries
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 26 for Kijun, 52 for Senkou B, 20 for volume, 14 for ATR, 50 for EMA)
    start = max(26, 52, 20, 14, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or \
           np.isnan(senkou_b[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine cloud color and position
        # Green cloud: Senkou A > Senkou B (bullish)
        # Red cloud: Senkou A < Senkou B (bearish)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # Ichimoku signals with 1d EMA filter and volume confirmation
        # Tenkan/Kijun cross
        tk_cross_up = tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]
        tk_cross_down = tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Long: TK cross up + price above cloud + price > 1d EMA + volume
        # Short: TK cross down + price below cloud + price < 1d EMA + volume
        long_signal = tk_cross_up and price_above_cloud and (close[i] > ema_1d_aligned[i]) and volume_filter[i]
        short_signal = tk_cross_down and price_below_cloud and (close[i] < ema_1d_aligned[i]) and volume_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or TK cross down
            if close[i] <= stop_price or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or TK cross up
            if close[i] >= stop_price or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

---  --- ---
---  --- ---