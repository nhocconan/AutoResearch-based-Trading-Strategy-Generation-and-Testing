# Strategy: 4h_1dCamarilla_R3S3_Breakout_1hEMA200_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.357 | +10.9% | -7.2% | 155 | FAIL |
| ETHUSDT | 0.837 | +52.2% | -4.8% | 110 | PASS |
| SOLUSDT | -0.019 | +18.0% | -13.6% | 106 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.686 | +12.6% | -4.1% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) with 1h EMA200 trend filter and volume spike confirmation
# Uses 1h HTF for EMA200 trend to avoid whipsaws and 1d HTF for Camarilla pivot calculation (key intraday support/resistance).
# Long when price breaks above 1d R3 in uptrend (1h close > 1h EMA200) with volume spike.
# Short when price breaks below 1d S3 in downtrend (1h close < 1h EMA200) with volume spike.
# Designed for low trade frequency (~20-30/year on 4h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with strict threshold (>3.0x average) to reduce overtrading.
# Stoploss at 2.0 * ATR and take profit at 1.5 * ATR to limit losing trades and secure gains.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at pivot levels.
# Focus on BTC/ETH as primary targets.

name = "4h_1dCamarilla_R3S3_Breakout_1hEMA200_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1h data ONCE before loop for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) using typical price
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tp_1d = typical_price_1d.values
    
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align 1d Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1h EMA(200) for trend filter
    close_1h = df_1h['close'].values
    ema_200_1h = pd.Series(close_1h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1h EMA to 4h timeframe
    ema_200_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_200_1h)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 200  # warmup for EMA(200)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 3.0x 50-period average (strict to reduce trades)
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (3.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema = ema_200_1h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d R3 with 1h uptrend (close > EMA200)
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d S3 with 1h downtrend (close < EMA200)
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 11:40
