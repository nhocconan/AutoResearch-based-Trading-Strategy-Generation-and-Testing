# 12h Donchian(15) breakout + daily trend filter + volume confirmation
# Strategy designed to capture medium-term trends with strict entry filters.
# Uses 12h timeframe to reduce noise and transaction costs.
# Entry requires: 1) Donchian breakout, 2) Daily EMA trend alignment, 3) Volume > 1.5x average
# Exit on opposite Donchian breakout or trend reversal.
# Expected trade count: 50-150 over 4 years (12-37/year) - within target range.

name = "12h_donchian15_1d_ema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA(50) for trend filter - calculated ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian(15) channels - tighter for fewer, higher-quality signals
    donchian_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR closes below EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR closes above EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA50 trend + volume confirmation
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i]:
                    # Breakout above Donchian high in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i]:
                    # Breakdown below Donchian low in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals