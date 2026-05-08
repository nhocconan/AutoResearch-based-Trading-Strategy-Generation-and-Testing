# 12h Williams Fractal Trend Strategy
# Hypothesis: Williams Fractals on weekly timeframe provide high-probability reversal zones
# Combined with 12h trend direction and volume confirmation for breakout/retest entries
# Works in bull markets (breakout of bullish fractals) and bear markets (breakdown of bearish fractals)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsFractal_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams Fractals (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Fractals (5-bar pattern)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    bearish_fractal = np.zeros(len(high_1w), dtype=bool)
    bullish_fractal = np.zeros(len(low_1w), dtype=bool)
    
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and
            high_1w[i-3] < high_1w[i-2] and
            high_1w[i+1] < high_1w[i-1]):
            bearish_fractal[i-1] = True
            
        # Bullish fractal
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and
            low_1w[i-3] > low_1w[i-2] and
            low_1w[i+1] > low_1w[i-1]):
            bullish_fractal[i-1] = True
    
    # Need 2-bar confirmation for fractals (per Williams)
    bearish_fractal_confirmed = np.zeros_like(bearish_fractal, dtype=bool)
    bullish_fractal_confirmed = np.zeros_like(bullish_fractal, dtype=bool)
    
    for i in range(len(bearish_fractal)):
        if bearish_fractal[i] and i + 2 < len(bearish_fractal):
            # Confirm if price goes below the fractal low within 2 bars
            if np.any(low_1w[i+1:i+3] < low_1w[i]):
                bearish_fractal_confirmed[i] = True
        if bullish_fractal[i] and i + 2 < len(bullish_fractal):
            # Confirm if price goes above the fractal high within 2 bars
            if np.any(high_1w[i+1:i+3] > high_1w[i]):
                bullish_fractal_confirmed[i] = True
    
    # Align confirmed fractals to 12h timeframe with 2-bar delay (as per rule)
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal_confirmed.astype(float), additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal_confirmed.astype(float), additional_delay_bars=2)
    
    # 12h EMA(21) for trend direction
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_up = ema_21 > np.roll(ema_21, 1)
    ema_up = np.where(np.isnan(ema_up), False, ema_up)
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bullish fractal breakout with volume and uptrend
            if bullish_aligned[i] > 0.5 and ema_up[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: bearish fractal breakdown with volume and downtrend
            elif bearish_aligned[i] > 0.5 and not ema_up[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: opposite fractal or trend change
            if bearish_aligned[i] > 0.5 or not ema_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite fractal or trend change
            if bullish_aligned[i] > 0.5 or ema_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals