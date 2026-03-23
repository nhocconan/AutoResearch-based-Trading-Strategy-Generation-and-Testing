# Strategy: mtf_12h_simplified_hma_donchian_1d_filter_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.370 | +11.6% | -7.3% | 207 | FAIL |
| ETHUSDT | -0.774 | -2.9% | -13.8% | 237 | FAIL |
| SOLUSDT | 0.185 | +29.8% | -15.4% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.117 | +7.1% | -5.2% | 82 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #026: 12h Simplified HMA-Donchian with 1d Trend Filter

Hypothesis: Previous 12h strategy (#012) failed due to overly complex entry conditions
(3+ confluence requirements). This simplifies to 2 core conditions:
1. 12h HMA trend direction (fast > slow for long, fast < slow for short)
2. Donchian(20) breakout in trend direction
3. 1d HMA(21) confirms major trend (price above for long bias, below for short bias)

Key changes from #012:
- Removed 1w filter (too restrictive, kills trade frequency)
- Reduced entry confluence from 3+ to 2 core conditions
- Added volatility-adjusted position sizing (smaller size when ATR high)
- Simplified stoploss: 2.5 ATR trailing (same but cleaner logic)
- Added minimum trade frequency safeguard (looser entry when no trades for 30 bars)

Why 12h works:
- Natural 20-50 trades/year (fee drag manageable)
- Filters noise from lower TFs
- Captures major moves without whipsaw

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 volatility-adjusted
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simplified_hma_donchian_1d_filter_atr_v1"
timeframe = "12h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_16 = calculate_hma(close, 16)  # Faster HMA for entry
    hma_12h_48 = calculate_hma(close, 48)  # Slower HMA for trend
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    bars_since_entry = 0
    last_trade_bar = -50  # Track last trade for frequency control
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === HMA TREND STRENGTH ===
        hma_slope_long = hma_12h_16[i] > hma_12h_16[i-1] if i > 0 else False
        hma_slope_short = hma_12h_16[i] < hma_12h_16[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_ok_long = rsi_14[i] < 75  # Don't long at extreme overbought
        rsi_ok_short = rsi_14[i] > 25  # Don't short at extreme oversold
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        # Reduce size when ATR is high (volatile market = more risk)
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)  # Keep in safe range
        
        # === ENTRY LOGIC (SIMPLIFIED - 2 core conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need trend + breakout (2 conditions)
        # Optional: daily bias confirmation
        long_score = 0
        if hma_bullish:
            long_score += 1
        if breakout_long:
            long_score += 1
        if daily_bullish:
            long_score += 0.5  # Bonus, not required
        if rsi_ok_long:
            long_score += 0.5  # Bonus, not required
        
        # Enter long if score >= 2.0 (trend + breakout required)
        if long_score >= 2.0 and hma_bullish and breakout_long:
            new_signal = current_size
        
        # SHORT ENTRY: Need trend + breakout (2 conditions)
        short_score = 0
        if hma_bearish:
            short_score += 1
        if breakout_short:
            short_score += 1
        if daily_bearish:
            short_score += 0.5  # Bonus, not required
        if rsi_ok_short:
            short_score += 0.5  # Bonus, not required
        
        # Enter short if score >= 2.0 (trend + breakout required)
        if short_score >= 2.0 and hma_bearish and breakout_short:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), loosen entry slightly
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            # Allow entry with just trend + RSI (no breakout required)
            if hma_bullish and daily_bullish and rsi_14[i] < 60:
                new_signal = current_size * 0.7  # Smaller size for weaker signal
            elif hma_bearish and daily_bearish and rsi_14[i] > 40:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if 12h HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                bars_since_entry = 0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                bars_since_entry = 0
                last_trade_bar = i
            else:
                # Same direction, maintain position
                bars_since_entry += 1
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 20:12
