# 4h_HighLow_Momentum_Signal
# Hypothesis: Use the 4h high-low range as a momentum indicator. When the current close
# is above the previous high, it signals upward momentum; when below the previous low,
# it signals downward momentum. This captures breakouts from recent ranges. Combined
# with volume confirmation and a 1-day EMA trend filter to avoid counter-trend trades.
# Works in both bull and bear markets by following momentum with trend alignment.
# Targets 20-50 trades per year to minimize fee drag.

name = "4h_HighLow_Momentum_Signal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(20) for trend filter
    daily_close = df_1d['close'].values
    ema20_1d = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum signal: close above previous high = long, below previous low = short
        long_signal = close[i] > high[i-1]
        short_signal = close[i] < low[i-1]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: daily EMA direction
        uptrend = ema20_1d_aligned[i] > ema20_1d_aligned[i-1]
        downtrend = ema20_1d_aligned[i] < ema20_1d_aligned[i-1]
        
        if position == 0:
            # Enter long: momentum up + volume confirmation + uptrend
            if long_signal and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: momentum down + volume confirmation + downtrend
            elif short_signal and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: momentum down or no volume confirmation or trend turns down
            if short_signal or not vol_confirm or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: momentum up or no volume confirmation or trend turns up
            if long_signal or not vol_confirm or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals