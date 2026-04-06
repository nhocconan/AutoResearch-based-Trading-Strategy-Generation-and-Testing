# 1h strategy for BTC/ETH/SOL USDT-M perpetual futures
# Combines 1d Supertrend trend filter with 1h RSI mean reversion entries
# Trend filter reduces false signals in ranging markets, RSI provides entry timing
# Target: 100-200 total trades over 4 years (25-50/year) to balance frequency and cost
# Works in bull/bear: Supertrend adapts to volatility, RSI mean reversion works in both regimes

name = "exp_12994_1h_supertrend_rsi_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    
    # Basic bands
    basic_upper = (high + low) / 2 + multiplier * atr
    basic_lower = (high + low) / 2 - multiplier * atr
    
    # Final bands
    final_upper = np.zeros_like(close)
    final_lower = np.zeros_like(close)
    
    for i in range(len(close)):
        if i == 0:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
        else:
            if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i-1]
                
            if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = final_lower[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] <= final_upper[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = 1
            else:
                if close[i] >= final_lower[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = -1
    
    return supertrend, direction

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Supertrend
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    supertrend_d, direction_d = calculate_supertrend(high_d, low_d, close_d, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    
    # Align Supertrend to hourly
    supertrend_aligned = align_htf_to_ltf(prices, df_daily, supertrend_d)
    direction_aligned = align_htf_to_ltf(prices, df_daily, direction_d)
    
    # Calculate hourly indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(SUPERTREND_PERIOD, RSI_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Supertrend not available
        if np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit on RSI overbought or trend change
            if rsi[i] >= RSI_OVERBOUGHT or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = SIGNAL_SIZE
                
        elif position == -1:  # short position
            # Exit on RSI oversold or trend change
            if rsi[i] <= RSI_OVERSOLD or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -SIGNAL_SIZE
                
        else:  # flat position
            # Entry conditions
            if direction_aligned[i] == 1 and rsi[i] <= RSI_OVERSOLD:
                # Long: uptrend + oversold
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif direction_aligned[i] == -1 and rsi[i] >= RSI_OVERBOUGHT:
                # Short: downtrend + overbought
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals