#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA (Kaufman Adaptive Moving Average) trend direction with RSI(14) overbought/oversold levels
# and Choppiness Index regime filter. KAMA adapts to market noise, reducing whipsaw in sideways markets.
# RSI provides mean-reversion signals at extremes, while Choppiness Index filters for ranging markets (CHOP > 61.8).
# This combination aims to capture mean-reversion in ranging markets and avoid trending markets where mean-reversion fails.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk via ATR stoploss.

name = "exp_13310_1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_FAST = 2    # Fast EMA constant for ER calculation
KAMA_ER_SLOW = 30   # Slow EMA constant for ER calculation
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # > 61.8 = ranging market
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_kama(close, er_fast, er_slow):
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(periods=10))  # 10-period net change
    volatility = close_s.diff().abs().rolling(window=10, min_periods=1).sum()
    er = direction / volatility.replace(0, 1e-10)  # Avoid division by zero
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1))**2
    kama = [close_s.iloc[0]]  # Initialize with first value
    for i in range(1, len(close_s)):
        kama.append(kama[-1] + sc.iloc[i] * (close_s.iloc[i] - kama[-1]))
    return np.array(kama)

def calculate_rsi(close, period):
    """Calculate Relative Strength Index (RSI)"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                               np.abs(low - np.roll(close, 1)))).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max()
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr.sum() / (high_roll - low_roll)) / np.log10(period)
    return chop.values

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
    
    # Load daily data for KAMA, RSI, Choppiness, Volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate indicators
    kama = calculate_kama(close, KAMA_ER_FAST, KAMA_ER_SLOW)
    rsi = calculate_rsi(close, RSI_PERIOD)
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(10, RSI_PERIOD, CHOPPINESS_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if any indicator not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Regime filter: only trade in ranging markets (Choppiness > threshold)
        ranging_market = chop[i] > CHOPPINESS_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean-reversion signals: RSI extremes in ranging markets
        if position == 0:
            if ranging_market and volume_ok:
                if rsi[i] < RSI_OVERSOLD and close[i] > kama[i]:  # Oversold but price above KAMA (bullish bias)
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif rsi[i] > RSI_OVERBOUGHT and close[i] < kama[i]:  # Overbought but price below KAMA (bearish bias)
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when RSI returns to neutral or price crosses below KAMA
            if rsi[i] >= 50 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when RSI returns to neutral or price crosses above KAMA
            if rsi[i] <= 50 or close[i] > kama[i]:
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

# Hypothesis: 1-day Donchian channel breakout with weekly EMA trend filter and volume confirmation.
# Donchian channels identify breakouts of recent highs/lows, capturing momentum.
# Weekly EMA ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation filters for institutional participation. ATR-based stoploss manages risk.
# Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support).
# Target: 75-200 total trades over 4 years (19-50/year).

name = "exp_13310_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20          # Weekly EMA
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
    
    # Calculate 1d indicators
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
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day RSI(14) with weekly trend filter and volume confirmation.
# RSI identifies overbought/oversold conditions for mean-reversion entries.
# Weekly EMA ensures alignment with higher timeframe trend to avoid counter-trend trades in strong trends.
# Volume confirmation filters for institutional participation. ATR-based stoploss manages risk.
# Works in ranging markets (RSI reversals) and avoids trending markets where mean-reversion fails.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "exp_13310_1d_rsi14_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_PERIOD = 20          # Weekly EMA
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate Relative Strength Index (RSI)"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Calculate 1d indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Mean-reversion signals: RSI extremes in trending markets (trade WITH trend)
        if position == 0:
            if volume_ok:
                if rsi[i] < RSI_OVERSOLD and uptrend:  # Oversold in uptrend -> long
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif rsi[i] > RSI_OVERBOUGHT and downtrend:  # Overbought in downtrend -> short
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when RSI returns to neutral or trend changes
            if rsi[i] >= 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when RSI returns to neutral or trend changes
            if rsi[i] <= 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

--- END OF FILE ---