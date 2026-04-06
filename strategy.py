#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour RSI(2) Extreme Reversion with Volume and Trend Filter.
# Uses 2-period RSI for extreme oversold/overbought conditions combined with:
# - Volume filter (current volume > 1.8x 20-period average)
# - 50-period EMA trend filter (long only above, short only below)
# - ATR-based stop loss and profit target at 2x ATR
# Designed to capture mean reversion moves in both bull and bear markets.
# Target: 75-150 trades over 4 years (19-38/year).

name = "6h_rsi2_extreme_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing (alpha = 1/period)
    alpha = 1.0 / 2
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) for trend filter
    ema50 = np.full(n, np.nan)
    ema50[0] = close[0]
    alpha_ema = 2.0 / (50 + 1)
    for i in range(1, n):
        ema50[i] = alpha_ema * close[i] + (1 - alpha_ema) * ema50[i-1]
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if np.isnan(ema50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit conditions
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            profit_target = entry_price + 2.0 * atr_approx
            
            if (rsi[i] >= 70 or  # RSI overbought exit
                close[i] <= stop_loss_level or
                close[i] >= profit_target):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            profit_target = entry_price - 2.0 * atr_approx
            
            if (rsi[i] <= 30 or  # RSI oversold exit
                close[i] >= stop_loss_level or
                close[i] <= profit_target):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with all filters
            if volume_filter:
                # Long setup: RSI < 10 (extremely oversold) + price above EMA50
                if (rsi[i] < 10 and close[i] > ema50[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short setup: RSI > 90 (extremely overbought) + price below EMA50
                elif (rsi[i] > 90 and close[i] < ema50[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour RSI(2) Extreme Reversion with Volume and Trend Filter.
# Uses 2-period RSI for extreme oversold/overbought conditions combined with:
# - Volume filter (current volume > 1.8x 20-period average)
# - 50-period EMA trend filter (long only above, short only below)
# - ATR-based stop loss and profit target at 2x ATR
# Designed to capture mean reversion moves in both bull and bear markets.
# Target: 75-150 trades over 4 years (19-38/year).

name = "6h_rsi2_extreme_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing (alpha = 1/period)
    alpha = 1.0 / 2
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) for trend filter
    ema50 = np.full(n, np.nan)
    ema50[0] = close[0]
    alpha_ema = 2.0 / (50 + 1)
    for i in range(1, n):
        ema50[i] = alpha_ema * close[i] + (1 - alpha_ema) * ema50[i-1]
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if np.isnan(ema50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit conditions
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            profit_target = entry_price + 2.0 * atr_approx
            
            if (rsi[i] >= 70 or  # RSI overbought exit
                close[i] <= stop_loss_level or
                close[i] >= profit_target):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            profit_target = entry_price - 2.0 * atr_approx
            
            if (rsi[i] <= 30 or  # RSI oversold exit
                close[i] >= stop_loss_level or
                close[i] <= profit_target):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with all filters
            if volume_filter:
                # Long setup: RSI < 10 (extremely oversold) + price above EMA50
                if (rsi[i] < 10 and close[i] > ema50[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short setup: RSI > 90 (extremely overbought) + price below EMA50
                elif (rsi[i] > 90 and close[i] < ema50[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals