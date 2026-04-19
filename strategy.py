# 4h_Camarilla_R1S1_Breakout_VolumeATRFilter_V1
# Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as institutional support/resistance.
# Breakouts above R1 or below S1 with volume confirmation indicate institutional participation.
# ATR filter avoids whipsaws in low volatility regimes. Works in bull/bear by following breakout direction.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1S1_Breakout_VolumeATRFilter_V1"
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
    
    # Get 1d data for Camarilla pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d volume average (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR for volatility filter and stop condition
    # Use 14-period ATR on 4h data
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first tr is infinite
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average ATR (avoid low vol chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_ma[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
            
        # Volatility and volume filters
        if not vol_filter[i] or not vol_confirm[i]:
            # No new positions in low vol/low vol confirmation, but maintain existing
            if position == 0:
                signals[i] = 0.0
                continue
            else:
                # Hold position but check exits below
                pass
        
        if position == 0:
            # Look for long entry: break above R1 with volume
            if close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short entry: break below S1 with volume
            elif close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on break below S1 or volatility collapse
            if close[i] < s1_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on break above R1 or volatility collapse
            if close[i] > r1_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals