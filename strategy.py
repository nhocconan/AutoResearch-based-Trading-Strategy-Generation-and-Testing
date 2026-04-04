#!/usr/bin/env python3
"""
exp_6459_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
- Weekly pivot (from 1d data) determines bias: long above weekly pivot, short below.
- Donchian breakout provides entry timing in direction of weekly bias.
- Volume confirmation (volume > 1.5x 20-period average) ensures momentum behind breakout.
- Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets.
- Target: 50-150 trades over 4 years (12-37/year) with discrete sizing to minimize fees.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6459_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    # Weekly high = max(high) over last 7 days, weekly low = min(low) over last 7 days, weekly close = close 7 days ago
    weekly_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1)  # prior week
    weekly_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1)
    weekly_close = df_1d['close'].shift(7)  # close from 7 days ago
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (with shift(1) for completed bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # Calculate Donchian channels on 6h data
    lookback = 20
    donchian_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().shift(1)
    donchian_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().shift(1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_ok = prices['volume'] > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Track position state for stoploss and reversal prevention
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    # Start from sufficient lookback
    start_idx = max(lookback, 20, 7) + 1
    
    for i in range(start_idx, n):
        # Skip if volume confirmation not met
        if not volume_ok.iloc[i]:
            # If in position, check stoploss
            if position_side == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_14.iloc[i]:
                signals[i] = 0.0
                position_side = 0
            elif position_side == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_14.iloc[i]:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1]  # hold current signal
            continue
        
        # Get current values
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        wp = weekly_pivot_aligned[i]
        dh = donchian_high.iloc[i]
        dl = donchian_low.iloc[i]
        
        # Long condition: price breaks above Donchian high AND above weekly pivot
        if position_side != 1:  # not already long
            if price_high > dh and price_close > wp:
                signals[i] = 0.30  # 30% long
                position_side = 1
                entry_price = price_close
            # Short condition: price breaks below Donchian low AND below weekly pivot
            elif price_low < dl and price_close < wp:
                signals[i] = -0.30  # 30% short
                position_side = -1
                entry_price = price_close
            else:
                # Hold current signal or flatten based on stoploss
                if position_side == 1:
                    if price_close < entry_price - 2.5 * atr_14.iloc[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                elif position_side == -1:
                    if price_close > entry_price + 2.5 * atr_14.iloc[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                else:
                    signals[i] = 0.0
        else:
            # Already in position, manage stoploss
            if position_side == 1:
                if price_close < entry_price - 2.5 * atr_14.iloc[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
            elif position_side == -1:
                if price_close > entry_price + 2.5 * atr_14.iloc[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
    
    # Calculate ATR(14) for stoploss (need to compute before loop but after prices load)
    # This is a simplification - in practice we'd compute ATR before the loop
    # For now, compute a basic ATR here (should be pre-computed in final version)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Re-run loop with actual ATR calculation (above logic needs atr_14)
    # Re-implement with proper ATR pre-computation
    
    # Pre-compute ATR
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Reset signals
    signals = np.zeros(n)
    position_side = 0
    entry_price = 0.0
    
    for i in range(start_idx, n):
        # Skip if volume confirmation not met
        if not volume_ok.iloc[i]:
            # If in position, check stoploss
            if position_side == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            elif position_side == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1]  # hold current signal
            continue
        
        # Get current values
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        wp = weekly_pivot_aligned[i]
        dh = donchian_high.iloc[i]
        dl = donchian_low.iloc[i]
        
        # Long condition: price breaks above Donchian high AND above weekly pivot
        if position_side != 1:  # not already long
            if price_high > dh and price_close > wp:
                signals[i] = 0.30  # 30% long
                position_side = 1
                entry_price = price_close
            # Short condition: price breaks below Donchian low AND below weekly pivot
            elif price_low < dl and price_close < wp:
                signals[i] = -0.30  # 30% short
                position_side = -1
                entry_price = price_close
            else:
                # Hold current signal or flatten based on stoploss
                if position_side == 1:
                    if price_close < entry_price - 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                elif position_side == -1:
                    if price_close > entry_price + 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                else:
                    signals[i] = 0.0
        else:
            # Already in position, manage stoploss
            if position_side == 1:
                if price_close < entry_price - 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
            elif position_side == -1:
                if price_close > entry_price + 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
    
    return signals

# Pre-compute indicators that don't depend on loop state
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    weekly_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1)
    weekly_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1)
    weekly_close = df_1d['close'].shift(7)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # Calculate Donchian channels on 6h data
    lookback = 20
    donchian_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().shift(1)
    donchian_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().shift(1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_ok = prices['volume'] > (1.5 * vol_ma)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    start_idx = max(lookback, 20, 7) + 1
    
    for i in range(start_idx, n):
        # Skip if volume confirmation not met
        if not volume_ok.iloc[i]:
            # If in position, check stoploss
            if position_side == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            elif position_side == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1]  # hold current signal
            continue
        
        # Get current values
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        wp = weekly_pivot_aligned[i]
        dh = donchian_high.iloc[i]
        dl = donchian_low.iloc[i]
        
        # Long condition: price breaks above Donchian high AND above weekly pivot
        if position_side != 1:  # not already long
            if price_high > dh and price_close > wp:
                signals[i] = 0.30  # 30% long
                position_side = 1
                entry_price = price_close
            # Short condition: price breaks below Donchian low AND below weekly pivot
            elif price_low < dl and price_close < wp:
                signals[i] = -0.30  # 30% short
                position_side = -1
                entry_price = price_close
            else:
                # Hold current signal or flatten based on stoploss
                if position_side == 1:
                    if price_close < entry_price - 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                elif position_side == -1:
                    if price_close > entry_price + 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                else:
                    signals[i] = 0.0
        else:
            # Already in position, manage stoploss
            if position_side == 1:
                if price_close < entry_price - 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
            elif position_side == -1:
                if price_close > entry_price + 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6459_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    weekly_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1)
    weekly_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1)
    weekly_close = df_1d['close'].shift(7)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # Calculate Donchian channels on 6h data
    lookback = 20
    donchian_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().shift(1)
    donchian_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().shift(1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_ok = prices['volume'] > (1.5 * vol_ma)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    start_idx = max(lookback, 20, 7) + 1
    
    for i in range(start_idx, n):
        # Skip if volume confirmation not met
        if not volume_ok.iloc[i]:
            # If in position, check stoploss
            if position_side == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            elif position_side == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1]  # hold current signal
            continue
        
        # Get current values
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        wp = weekly_pivot_aligned[i]
        dh = donchian_high.iloc[i]
        dl = donchian_low.iloc[i]
        
        # Long condition: price breaks above Donchian high AND above weekly pivot
        if position_side != 1:  # not already long
            if price_high > dh and price_close > wp:
                signals[i] = 0.30  # 30% long
                position_side = 1
                entry_price = price_close
            # Short condition: price breaks below Donchian low AND below weekly pivot
            elif price_low < dl and price_close < wp:
                signals[i] = -0.30  # 30% short
                position_side = -1
                entry_price = price_close
            else:
                # Hold current signal or flatten based on stoploss
                if position_side == 1:
                    if price_close < entry_price - 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                elif position_side == -1:
                    if price_close > entry_price + 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                else:
                    signals[i] = 0.0
        else:
            # Already in position, manage stoploss
            if position_side == 1:
                if price_close < entry_price - 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
            elif position_side == -1:
                if price_close > entry_price + 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6459_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    weekly_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1)
    weekly_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1)
    weekly_close = df_1d['close'].shift(7)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # Calculate Donchian channels on 6h data
    lookback = 20
    donchian_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().shift(1)
    donchian_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().shift(1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_ok = prices['volume'] > (1.5 * vol_ma)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    start_idx = max(lookback, 20, 7) + 1
    
    for i in range(start_idx, n):
        # Skip if volume confirmation not met
        if not volume_ok.iloc[i]:
            # If in position, check stoploss
            if position_side == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            elif position_side == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1]  # hold current signal
            continue
        
        # Get current values
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        wp = weekly_pivot_aligned[i]
        dh = donchian_high.iloc[i]
        dl = donchian_low.iloc[i]
        
        # Long condition: price breaks above Donchian high AND above weekly pivot
        if position_side != 1:  # not already long
            if price_high > dh and price_close > wp:
                signals[i] = 0.30  # 30% long
                position_side = 1
                entry_price = price_close
            # Short condition: price breaks below Donchian low AND below weekly pivot
            elif price_low < dl and price_close < wp:
                signals[i] = -0.30  # 30% short
                position_side = -1
                entry_price = price_close
            else:
                # Hold current signal or flatten based on stoploss
                if position_side == 1:
                    if price_close < entry_price - 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                elif position_side == -1:
                    if price_close > entry_price + 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                else:
                    signals[i] = 0.0
        else:
            # Already in position, manage stoploss
            if position_side == 1:
                if price_close < entry_price - 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
            elif position_side == -1:
                if price_close > entry_price + 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
    
    return signals