#!/usr/bin/env python3
"""
6h VWAP Reversion + Volume Spike + Trend Filter
Hypothesis: Price tends to revert to VWAP during high volatility spikes when aligned with higher timeframe trend.
Works in bull/bear: mean reversion in ranging markets, trend following when price sustains above/below VWAP.
Targets 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14355_6h_vwap_reversion_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # VWAP deviation (percentage)
    vwap_dev = (close - vwap) / vwap * 100.0
    
    # Volume spike detector (volume > 2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # ATR for dynamic thresholds and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start = 50  # For EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap) or np.isnan(vwap_dev[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to VWAP OR trend fails OR stoploss
            if (vwap_dev[i] >= -0.5 or  # Returned to VWAP (within 0.5%)
                close[i] < ema50_1d_aligned[i] or  # Trend failed
                close[i] <= entry_price - 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to VWAP OR trend fails OR stoploss
            if (vwap_dev[i] <= 0.5 or   # Returned to VWAP (within 0.5%)
                close[i] > ema50_1d_aligned[i] or  # Trend failed
                close[i] >= entry_price + 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VWAP deviation + volume spike + trend alignment
            long_setup = (vwap_dev[i] <= -2.0 and  # Price 2%+ below VWAP
                          vol_spike[i] and         # Volume spike
                          close[i] > ema50_1d_aligned[i])  # Uptrend
            
            short_setup = (vwap_dev[i] >= 2.0 and   # Price 2%+ above VWAP
                           vol_spike[i] and         # Volume spike
                           close[i] < ema50_1d_aligned[i])  # Downtrend
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Volume-Weighted RSI + Trend Filter
Hypothesis: Combines volume-weighted RSI for momentum with higher timeframe trend filter.
Volume weighting gives more importance to price moves on high volume, reducing false signals.
Works in bull/bear: RSI extremes in ranging markets, trend following when aligned with daily trend.
Targets 70-140 total trades over 4 years (18-35/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14355_6h_vw_rsi_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA100 for trend filter (smoother than 50)
    ema100_1d = pd.Series(close_1d).ewm(span=100, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume-weighted RSI calculation
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Smoothed volume-weighted gains/losses (Wilder's smoothing)
    avg_vg = pd.Series(vol_gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_vl = pd.Series(vol_loss).ewm(alpha=1/14, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vl != 0, avg_vg / avg_vl, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require at least 150% of average volume
    
    # ATR for dynamic thresholds and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start = 100  # For EMA100
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(vw_rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI returns to neutral OR trend fails OR stoploss
            if (vw_rsi[i] >= 50 or                  # RSI returned to neutral
                close[i] < ema100_1d_aligned[i] or  # Trend failed
                close[i] <= entry_price - 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI returns to neutral OR trend fails OR stoploss
            if (vw_rsi[i] <= 50 or                  # RSI returned to neutral
                close[i] > ema100_1d_aligned[i] or  # Trend failed
                close[i] >= entry_price + 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VW-RSI extremes + volume filter + trend alignment
            long_setup = (vw_rsi[i] <= 30 and       # Oversold
                          vol_filter[i] and         # Volume confirmation
                          close[i] > ema100_1d_aligned[i])  # Uptrend
            
            short_setup = (vw_rsi[i] >= 70 and      # Overbought
                           vol_filter[i] and        # Volume confirmation
                           close[i] < ema100_1d_aligned[i])  # Downtrend
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Price Channel Breakout + Volume Confirmation + Volatility Filter
Hypothesis: Price breaks out of dynamic channels (Keltner) during low volatility periods 
with volume confirmation. Works in bull/bear: breakouts capture new trends, volatility filter 
avoids false breakouts in choppy markets. Targets 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14355_6h_keltner_breakout_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel components
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # EMA of typical price (20-period)
    ema_tp = pd.Series(typical_price).ewm(span=20, min_periods=20).mean().values
    
    # ATR (10-period for tighter bands)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Bands (EMA ± 1.5 * ATR)
    upper_band = ema_tp + (1.5 * atr)
    lower_band = ema_tp - (1.5 * atr)
    
    # Volume spike detector (volume > 1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    # Volatility filter: avoid high volatility periods (ATR ratio)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr <= (1.2 * atr_ma)  # Only trade when volatility is below 1.2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start = 50  # For EMA and ATR MA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(ema_tp[i]) or np.isnan(upper_band[i]) or \
           np.isnan(lower_band[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to EMA OR trend fails OR stoploss
            if (close[i] <= ema_tp[i] or                  # Returned to midpoint
                close[i] < ema50_1w_aligned[i] or         # Trend failed
                close[i] <= entry_price - 2.0 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to EMA OR trend fails OR stoploss
            if (close[i] >= ema_tp[i] or                  # Returned to midpoint
                close[i] > ema50_1w_aligned[i] or         # Trend failed
                close[i] >= entry_price + 2.0 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: band break + volume spike + low volatility + trend alignment
            long_setup = (close[i] > upper_band[i] and    # Break above upper band
                          vol_spike[i] and                # Volume spike
                          vol_filter[i] and               # Low volatility filter
                          close[i] > ema50_1w_aligned[i]) # Uptrend
            
            short_setup = (close[i] < lower_band[i] and   # Break below lower band
                           vol_spike[i] and               # Volume spike
                           vol_filter[i] and              # Low volatility filter
                           close[i] < ema50_1w_aligned[i]) # Downtrend
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Adaptive Supertrend + Volume Confirmation
Hypothesis: Supertrend adapts to volatility with ATR multiplier based on volatility regime.
Volume confirmation filters false signals. Works in bull/bear: adapts to changing volatility 
regimes, volume confirms institutional participation. Targets 65-130 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14355_6h_adaptive_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR calculation (10-period for responsiveness)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # ATR-based volatility regime detector
    atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr / atr_ma  # Current ATR relative to average
    
    # Adaptive ATR multiplier: lower in high vol, higher in low vol
    # Clip ratio between 0.5 and 2.0 for stability
    atr_ratio_clipped = np.clip(atr_ratio, 0.5, 2.0)
    # Invert so multiplier decreases as volatility increases
    adaptive_multiplier = 3.0 * (2.0 - atr_ratio_clipped)  # Range: 3.0 to 1.5
    
    # Supertrend calculation
    hl2 = (high + low) / 2.0
    
    # Upper and Lower Bands
    upper_band = hl2 + (adaptive_multiplier * atr)
    lower_band = hl2 - (adaptive_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    # First value
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, n):
        # Calculate bands
        upper = hl2[i] + (adaptive_multiplier[i] * atr[i])
        lower = hl2[i] - (adaptive_multiplier[i] * atr[i])
        
        # Determine Supertrend value
        if supertrend[i-1] <= upper_band[i-1]:
            if close[i] > upper:
                supertrend[i] = upper
                direction[i] = -1  # Downtrend
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
        else:
            if close[i] < lower:
                supertrend[i] = lower
                direction[i] = 1   # Uptrend
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
        
        # Store bands for next iteration
        upper_band[i] = upper
        lower_band[i] = lower
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)  # Require at least 130% of average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start = 30  # For ATR MA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(supertrend[i]) or np.isnan(direction[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: Supertrend turns bearish OR trend fails OR stoploss
            if (direction[i] == -1 or                  # Supertrend bearish
                close[i] < ema100_1d_aligned[i] or     # Trend failed
                close[i] <= entry_price - 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Supertrend turns bullish OR trend fails OR stoploss
            if (direction[i] == 1 or                   # Supertrend bullish
                close[i] > ema100_1d_aligned[i] or     # Trend failed
                close[i] >= entry_price + 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Supertrend signal + volume filter + trend alignment
            long_setup = (direction[i] == 1 and      # Supertrend uptrend
                          vol_filter[i] and          # Volume confirmation
                          close[i] > ema100_1d_aligned[i])  # Additional trend filter
            
            short_setup = (direction[i] == -1 and    # Supertrend downtrend
                           vol_filter[i] and         # Volume confirmation
                           close[i] < ema100_1d_aligned[i])  # Additional trend filter
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout + ADX Trend Strength + Volume Filter
Hypothesis: Donchian channel breakouts filtered by ADX trend strength and volume.
Works in bull/bear: breakouts capture new trends, ADX ensures strong trends, volume confirms.
Targets 70-140 total trades over 4 years (18-35/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14355_6h_donchian_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14).mean().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.6 * vol_ma)  # Require at least 160% of average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to midpoint OR trend weakens OR stoploss
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] <= midpoint or                  # Returned to midpoint
                adx[i] < 20 or                           # Trend weakened
                close[i] <= entry_price - 2.5 * (highest_high[i] - lowest_low[i]) * 0.15):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midpoint OR trend weakens OR stoploss
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] >= midpoint or                  # Returned to midpoint
                adx[i] < 20 or                           # Trend weakened
                close[i] >= entry_price + 2.5 * (highest_high[i] - lowest_low[i]) * 0.15):  # Stoploss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + ADX strength + volume filter + trend alignment
            long_setup = (close[i] > highest_high[i] and   # Break above upper band
                          adx[i] >= 25 and                 # Strong trend
                          vol_filter[i] and                # Volume confirmation
                          close[i] > ema50_1w_aligned[i])  # Uptrend filter
            
            short_setup = (close[i] < lowest_low[i] and    # Break below lower band
                           adx[i] >= 25 and                # Strong trend
                           vol_filter[i] and               # Volume confirmation
                           close[i] < ema50_1w_aligned[i]) # Downtrend filter
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Elder Ray Index + Market Facilitation Index + Volume Filter
Hypothesis: Combines Elder Ray (bull/bear power) with MFI for volume-weighted momentum.
Works in bull/bear: Elder Ray shows power of bulls/bears, MFI confirms with volume.
Targets 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14355_6h_elder_ray_mfi_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Index components
    # Exponential Moving Average (13-period)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Market Facilitation Index (MFI) - (High - Low) / Volume
    # Avoid division by zero
    hl_range = high - low
    mfi = np.where(volume != 0, hl_range / volume, 0)
    
    # MFI smoothed (10-period average)
    mfi_smooth = pd.Series(mfi).rolling(window=10, min_periods=10).mean().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd