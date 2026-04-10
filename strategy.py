#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ATR-based stoploss
# - Long when price breaks above Camarilla H3 level AND 1w volume > 1.5x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level AND 1w volume > 1.5x 20-period volume SMA
# - Exit: ATR-based trailing stop (3x ATR from extreme) or Camarilla pivot reversion (H4/L4)
# - Position sizing: 0.25 discrete level to balance return and drawdown
# - Target: 15-30 trades/year on 1d timeframe to stay within fee drag limits
# - Uses Camarilla pivots for structure, volume for confirmation, ATR for risk management
# - Works in both bull (breakouts) and bear (mean reversion at pivots) markets

name = "1d_camarilla_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels (based on previous day)
    # H4, H3, H2, H1, L1, L2, L3, L4
    camarilla_h4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's pivots
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        pivot = (phigh + plow + pclose) / 3.0
        range_ = phigh - plow
        
        camarilla_h4[i] = pclose + range_ * 1.1 / 2.0
        camarilla_h3[i] = pclose + range_ * 1.1 / 4.0
        camarilla_l3[i] = pclose - range_ * 1.1 / 4.0
        camarilla_l4[i] = pclose - range_ * 1.1 / 2.0
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 1w volume SMA for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Track extreme prices for trailing stop
    long_extreme = np.full(n, np.nan)
    short_extreme = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after ATR and volume SMA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1w volume > 1.5x 20-period volume SMA
        # Each 1w bar = 5 1d bars (approx)
        idx_1w = i // 5
        if idx_1w < len(volume_1w):
            vol_confirm = volume_1w[idx_1w] > 1.5 * volume_sma_20_1w_aligned[i]
        else:
            vol_confirm = False
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3[i]  # Break below L3
        
        # Update extremes for trailing stop
        if position == 1:  # Long position
            if np.isnan(long_extreme[i-1]):
                long_extreme[i] = close[i]
            else:
                long_extreme[i] = max(long_extreme[i-1], close[i])
        elif position == -1:  # Short position
            if np.isnan(short_extreme[i-1]):
                short_extreme[i] = close[i]
            else:
                short_extreme[i] = min(short_extreme[i-1], close[i])
        else:
            long_extreme[i] = np.nan
            short_extreme[i] = np.nan
        
        # ATR-based trailing stop conditions
        stop_long = False
        stop_short = False
        
        if position == 1 and not np.isnan(long_extreme[i]):
            stop_long = close[i] < long_extreme[i] - 3.0 * atr[i]
        elif position == -1 and not np.isnan(short_extreme[i]):
            stop_short = close[i] > short_extreme[i] + 3.0 * atr[i]
        
        # Camarilla pivot reversion exit (H4/L4 levels)
        exit_long = close[i] < camarilla_h4[i]
        exit_short = close[i] > camarilla_l4[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if stop_long or exit_long:
                position = 0
                signals[i] = 0.0
                long_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if stop_short or exit_short:
                position = 0
                signals[i] = 0.0
                short_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = -0.25
    
    return signals