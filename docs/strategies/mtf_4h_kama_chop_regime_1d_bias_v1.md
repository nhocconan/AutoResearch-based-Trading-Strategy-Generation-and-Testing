# Strategy: mtf_4h_kama_chop_regime_1d_bias_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.479 | -3.9% | -6.1% | 314 | FAIL |
| ETHUSDT | -0.457 | +4.7% | -8.9% | 231 | FAIL |
| SOLUSDT | 0.305 | +39.2% | -28.1% | 222 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.056 | +6.4% | -6.3% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #029: 4h KAMA Trend + Choppiness Regime + 1d Bias

Hypothesis: Previous HMA-based strategies failed due to whipsaw in choppy markets.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - slower in chop, faster
in trends. Combined with Choppiness Index regime detection:
1. CHOP > 61.8 = range market → mean reversion entries (RSI extremes)
2. CHOP < 38.2 = trending market → trend follow entries (KAMA crossover)
3. 1d KAMA for major trend bias (prevents counter-trend trades)
4. ATR(14) trailing stoploss at 2.5x
5. Discrete position sizing (0.25-0.30) to minimize fee churn

Why this should work:
- KAMA adapts to market conditions (proven in literature)
- Choppiness Index filters regime (ETH Sharpe +0.923 in exp history)
- 4h timeframe = natural 20-50 trades/year target
- Simpler entry thresholds = more trades (avoiding 0-trade failures)
- 1d bias prevents major counter-trend losses (2022 crash protection)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_regime_1d_bias_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    Adapts to volatility: fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=period).sum()
    
    er = change / volatility
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    # Choppiness formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50).values
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_4h_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    hma_200 = calculate_hma(close, 200)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_4h_fast[i]) or np.isnan(kama_4h_slow[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > kama_1d_21_aligned[i]
        daily_bearish = close[i] < kama_1d_21_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_4h_fast[i] > kama_4h_slow[i]
        kama_bearish = kama_4h_fast[i] < kama_4h_slow[i]
        
        # === CHOPPINNESS REGIME ===
        choppy_market = chop_14[i] > 61.8
        trending_market = chop_14[i] < 38.2
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if daily_bullish:  # Only long when 1d bias is bullish
            if trending_market and kama_bullish:
                # Trend follow: KAMA crossover + RSI confirmation
                if 45 <= rsi_14[i] <= 65:
                    new_signal = current_size
            elif choppy_market:
                # Mean reversion: RSI oversold in range
                if rsi_14[i] < 35 and close[i] > hma_200[i]:
                    new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        elif daily_bearish:  # Only short when 1d bias is bearish
            if trending_market and kama_bearish:
                # Trend follow: KAMA crossover + RSI confirmation
                if 35 <= rsi_14[i] <= 55:
                    new_signal = -current_size
            elif choppy_market:
                # Mean reversion: RSI overbought in range
                if rsi_14[i] > 65 and close[i] < hma_200[i]:
                    new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~5 days on 4h), force entry with weaker signal
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if daily_bullish and kama_bullish and rsi_14[i] > 40:
                new_signal = current_size * 0.5
            elif daily_bearish and kama_bearish and rsi_14[i] < 60:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and daily_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
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
2026-03-22 21:05
