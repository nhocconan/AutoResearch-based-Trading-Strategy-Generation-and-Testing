# 12h Weekly Donchian Breakout with Volume Confirmation and Trend Filter
# Hypothesis: Price breaking weekly Donchian channels with volume confirmation and trend filter
# (price vs 200 EMA) works in both bull and bear markets by capturing strong momentum
# while avoiding whipsaws. In bull markets: buy breakouts above upper band. In bear markets:
# sell breakdowns below lower band. Weekly timeframe provides robust levels; 12h execution
# improves timing. Target: 15-35 trades/year (60-140 over 4 years).

name = "12h_weekly_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate upper and lower bands
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    upper = np.roll(upper, 1)
    lower = np.roll(lower, 1)
    
    # Handle first element
    if len(upper) > 1:
        upper[0] = upper[1]
        lower[0] = lower[1]
    else:
        upper[0] = 0
        lower[0] = 0
    
    # Align to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: breakdown below lower band or trend failure
            if low[i] <= lower_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: breakout above upper band or trend failure
            if high[i] >= upper_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper band with volume and trend alignment
            if high[i] > upper_aligned[i] and close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower band with volume and trend alignment
            elif low[i] < lower_aligned[i] and close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals