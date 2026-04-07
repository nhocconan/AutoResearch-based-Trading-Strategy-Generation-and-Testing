# 2024-06-01: 6h Camarilla Pivot + Volume + Trend Strategy
# Hypothesis: Camarilla pivot levels from 1d provide high-probability reversal/breakout zones.
# Fade at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets.
# Volume confirms institutional participation. Trend filter (12h EMA) avoids counter-trend trades.
# Designed for low frequency: target 15-30 trades/year per side to minimize fee drag.
# Works in bull markets (buy R4 breakouts) and bear markets (sell S4 breakdowns).

name = "6h_camarilla_pivot_1d_volume_trend_v1"
timeframe = "6h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Where C = close, H = high, L = low of previous day
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate pivot levels
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6h timeframe (these levels are valid for the entire day)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA/volume warmup
        # Skip if required data not available
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: return to R3/S3 or opposite extreme
        exit_long = close[i] < R3_aligned[i]  # Return to R3
        exit_short = close[i] > S3_aligned[i]  # Return to S3
        
        if position == 1:  # Long position
            # Exit on return to R3 or trend reversal
            if exit_long or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on return to S3 or trend reversal
            if exit_short or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: break above R4 + uptrend + volume confirmation
            if close[i] > R4_aligned[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: break below S4 + downtrend + volume confirmation
            elif close[i] < S4_aligned[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals