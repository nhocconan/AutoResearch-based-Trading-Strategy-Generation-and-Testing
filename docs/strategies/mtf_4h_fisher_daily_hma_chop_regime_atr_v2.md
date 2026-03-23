# Strategy: mtf_4h_fisher_daily_hma_chop_regime_atr_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.798 | -15.9% | -31.5% | 907 | FAIL |
| ETHUSDT | -0.252 | +1.8% | -22.7% | 929 | FAIL |
| SOLUSDT | 0.402 | +58.1% | -31.0% | 935 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.435 | +13.6% | -19.7% | 312 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #484: 4h Fisher Transform + Daily HMA Bias + Choppiness Regime + ATR Stop
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets 
(2025 test period). Combined with Daily HMA for trend bias and Choppiness Index for 
regime detection, we adapt strategy: mean-revert in choppy markets, trend-follow in 
trending markets. 4h timeframe balances noise reduction with reasonable trade frequency.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25) controls 
drawdown. 2.5*ATR stoploss appropriate for 4h bars.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_chop_regime_atr_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for identifying reversal points in bear/range markets.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate (high + low) / 2
        hl2 = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        
        # Normalize to -1 to +1 range
        highest_hl = max(high[i-period+1:i+1])
        lowest_hl = min(low[i-period+1:i+1])
        
        if highest_hl - lowest_hl > 0:
            value = 0.66 * ((hl2 - lowest_hl) / (highest_hl - lowest_hl) - 0.5) + 0.67 * (fisher[i-1] if i > period else 0)
            value = np.clip(value, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
            
            # Trigger line (previous fisher)
            trigger[i] = fisher[i-1] if i > period else fisher[i]
        else:
            fisher[i] = fisher[i-1] if i > period else 0
            trigger[i] = trigger[i-1] if i > period else 0
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_h = high[i-period+1:i+1].max()
        lowest_l = low[i-period+1:i+1].min()
        
        if highest_h - lowest_l > 0:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_h - lowest_l)) / np.log10(period)
        else:
            chop[i] = chop[i-1] if i > period else 50
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # 4h HMA for trend
    hma_4h = calculate_hma(close, 21)
    hma_4h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h HMA trend
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        hma_rising = hma_4h[i] > hma_4h[i-1] if i > 0 else False
        hma_falling = hma_4h[i] < hma_4h[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_4h_fast[i] > hma_4h[i]
        fast_below_slow = hma_4h_fast[i] < hma_4h[i]
        
        # Choppiness regime
        is_choppy = chop[i] > 55  # Range market
        is_trending = chop[i] < 45  # Trend market
        
        # Fisher Transform signals
        fisher_bullish = fisher[i] > fisher_trigger[i] and fisher[i] < -0.5
        fisher_bearish = fisher[i] < fisher_trigger[i] and fisher[i] > 0.5
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 else False
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending regime + Daily bullish + Fisher cross up + RSI ok
        if is_trending and daily_bullish and fisher_cross_up and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # Path 2: Choppy regime + Fisher oversold + RSI oversold (mean reversion)
        elif is_choppy and fisher[i] < -1.0 and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: Daily bullish + 4h HMA bullish + Fast HMA crossover up
        elif daily_bullish and hma_4h_bullish and fast_above_slow and hma_4h_fast[i] > hma_4h_fast[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 4: Fisher cross up from extreme + Daily not bearish
        elif fisher_cross_up and fisher[i-1] < -1.5 and not daily_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h HMA rising + RSI pullback to 40-50 + Daily bullish
        elif hma_rising and rsi[i] > 40 and rsi[i] < 52 and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending regime + Daily bearish + Fisher cross down + RSI ok
        if is_trending and daily_bearish and fisher_cross_down and rsi[i] > 40:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Choppy regime + Fisher overbought + RSI overbought (mean reversion)
        elif is_choppy and fisher[i] > 1.0 and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Daily bearish + 4h HMA bearish + Fast HMA crossover down
        elif daily_bearish and hma_4h_bearish and fast_below_slow and hma_4h_fast[i] < hma_4h_fast[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Fisher cross down from extreme + Daily not bullish
        elif fisher_cross_down and fisher[i-1] > 1.5 and not daily_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h HMA falling + RSI rally to 48-60 + Daily bearish
        elif hma_falling and rsi[i] > 48 and rsi[i] < 60 and daily_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 07:19
