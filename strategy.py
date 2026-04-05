#!/usr/bin/env python3
"""
exp_7519_6d_1d_12h_adx_rsi_v1
Hypothesis: 6d ADX filter with 1d RSI mean reversion and 12h trend confirmation. 
In trending markets (ADX > 25): trade pullbacks in trend direction using 1d RSI extremes. 
In ranging markets (ADX < 20): fade at Bollinger Band extremes. 
Uses 12h EMA200 for trend direction filter to avoid counter-trend trades. 
Targets 100-200 trades over 4 years (25-50/year) with balanced long/short.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7519_6d_1d_12h_adx_rsi_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BB_PERIOD = 20
BB_STD = 2.0
EMA_TREND = 200
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_adx(high, low, close, period):
    """Calculate ADX indicator"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d RSI for mean reversion
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Bollinger Bands for ranging markets
    sma_1d = pd.Series(close_1d).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    std_1d = pd.Series(close_1d).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    upper_bb = sma_1d + BB_STD * std_1d
    lower_bb = sma_1d - BB_STD * std_1d
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 12h EMA200 for trend direction
    close_12h = df_12h['close'].values
    ema_12h_200 = pd.Series(close_12h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_12h_200_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6d RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, RSI_PERIOD, BB_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema_12h_200_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        trending = adx_1d_aligned[i] > ADX_TREND_THRESHOLD
        ranging = adx_1d_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Trend direction from 12h EMA200
        uptrend = close[i] > ema_12h_200_aligned[i]
        downtrend = close[i] < ema_12h_200_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if trending:
            # In trending markets: trade pullbacks in trend direction
            long_entry = (
                uptrend and                    # 12h uptrend
                rsi_1d_aligned[i] < RSI_OVERSOLD and  # 1d RSI oversold
                rsi[i] < RSI_OVERSOLD          # 6d RSI oversold (timing)
            )
            short_entry = (
                downtrend and                  # 12h downtrend
                rsi_1d_aligned[i] > RSI_OVERBOUGHT and  # 1d RSI overbought
                rsi[i] > RSI_OVERBOUGHT        # 6d RSI overbought (timing)
            )
        elif ranging:
            # In ranging markets: fade at Bollinger Band extremes
            long_entry = (
                close[i] <= lower_bb_aligned[i] and  # at/below lower BB
                rsi[i] < RSI_OVERSOLD                # additional RSI filter
            )
            short_entry = (
                close[i] >= upper_bb_aligned[i] and  # at/above upper BB
                rsi[i] > RSI_OVERBOUGHT              # additional RSI filter
            )
        
        # Exit conditions
        long_exit = rsi[i] > 50  # exit when RSI returns to neutral
        short_exit = rsi[i] < 50  # exit when RSI returns to neutral
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7519_6d_1d_12h_adx_rsi_v1
Hypothesis: 6d ADX filter with 1d RSI mean reversion and 12h trend confirmation. 
In trending markets (ADX > 25): trade pullbacks in trend direction using 1d RSI extremes. 
In ranging markets (ADX < 20): fade at Bollinger Band extremes. 
Uses 12h EMA200 for trend direction filter to avoid counter-trend trades. 
Targets 100-200 trades over 4 years (25-50/year) with balanced long/short.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7519_6d_1d_12h_adx_rsi_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BB_PERIOD = 20
BB_STD = 2.0
EMA_TREND = 200
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_adx(high, low, close, period):
    """Calculate ADX indicator"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d RSI for mean reversion
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Bollinger Bands for ranging markets
    sma_1d = pd.Series(close_1d).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    std_1d = pd.Series(close_1d).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    upper_bb = sma_1d + BB_STD * std_1d
    lower_bb = sma_1d - BB_STD * std_1d
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 12h EMA200 for trend direction
    close_12h = df_12h['close'].values
    ema_12h_200 = pd.Series(close_12h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_12h_200_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6d RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, RSI_PERIOD, BB_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema_12h_200_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        trending = adx_1d_aligned[i] > ADX_TREND_THRESHOLD
        ranging = adx_1d_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Trend direction from 12h EMA200
        uptrend = close[i] > ema_12h_200_aligned[i]
        downtrend = close[i] < ema_12h_200_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if trending:
            # In trending markets: trade pullbacks in trend direction
            long_entry = (
                uptrend and                    # 12h uptrend
                rsi_1d_aligned[i] < RSI_OVERSOLD and  # 1d RSI oversold
                rsi[i] < RSI_OVERSOLD          # 6d RSI oversold (timing)
            )
            short_entry = (
                downtrend and                  # 12h downtrend
                rsi_1d_aligned[i] > RSI_OVERBOUGHT and  # 1d RSI overbought
                rsi[i] > RSI_OVERBOUGHT        # 6d RSI overbought (timing)
            )
        elif ranging:
            # In ranging markets: fade at Bollinger Band extremes
            long_entry = (
                close[i] <= lower_bb_aligned[i] and  # at/below lower BB
                rsi[i] < RSI_OVERSOLD                # additional RSI filter
            )
            short_entry = (
                close[i] >= upper_bb_aligned[i] and  # at/above upper BB
                rsi[i] > RSI_OVERBOUGHT              # additional RSI filter
            )
        
        # Exit conditions
        long_exit = rsi[i] > 50  # exit when RSI returns to neutral
        short_exit = rsi[i] < 50  # exit when RSI returns to neutral
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals