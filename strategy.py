# 12h_TurtleTrend_1dVolatilityBreakout
# Hypothesis: 12h Turtle-style breakout with 1d volatility filter to avoid false breakouts in choppy markets.
# Long when price breaks above 12h Donchian(20) high AND 1d ATR ratio > 0.8 (sufficient volatility).
# Short when price breaks below 12h Donchian(20) low AND 1d ATR ratio > 0.8.
# Exit when price crosses back inside the Donchian channel.
# Uses 12h timeframe for signals, 1d ATR for volatility regime filter.
# Designed to work in both bull and bear markets by capturing strong moves when volatility is present.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.

name = "12h_TurtleTrend_1dVolatilityBreakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for volatility filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Donchian(20) on 12h data
    donchian_period = 20
    upper_dc = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Daily ATR for volatility filter
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # ATR(20) - using 20-period for stability
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: current ATR / 50-period ATR average to detect volatility expansion
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma50
    atr_ratio[np.isnan(atr_ratio)] = 0  # Handle division by zero or NaN
    
    # Volatility filter: ATR ratio > 0.8 (avoid trading in extremely low volatility)
    vol_filter = atr_ratio > 0.8
    vol_filter_aligned = align_htf_to_ltf(prices, df_d, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 50)  # Sufficient warmup for ATR ratio
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND sufficient volatility
            long_cond = (close[i] > upper_dc[i]) and vol_filter_aligned[i]
            # Short conditions: price breaks below Donchian lower AND sufficient volatility
            short_cond = (close[i] < lower_dc[i]) and vol_filter_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals