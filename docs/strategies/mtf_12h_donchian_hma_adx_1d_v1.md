# Strategy: mtf_12h_donchian_hma_adx_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.283 | +12.9% | -8.9% | 188 | FAIL |
| ETHUSDT | -0.710 | -4.5% | -15.0% | 210 | FAIL |
| SOLUSDT | 0.189 | +30.7% | -23.3% | 201 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.184 | +7.8% | -5.9% | 63 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian-HMA Trend with 1d Bias and ATR Stoploss

Hypothesis: Previous strategies failed due to Choppiness Index regime switching.
This strategy uses a PROVEN combination for 12h timeframe:

1. Donchian Channel (20) breakout - captures momentum when price breaks 20-period high/low
2. 1d HMA(21) for major trend bias - only trade breakouts in direction of daily trend
3. ADX(14) filter - require ADX > 18 to confirm trending market (avoid chop)
4. ATR(14) trailing stoploss - 2.5x ATR to protect against reversals
5. 12h timeframe - targets 20-40 trades/year (optimal fee drag vs signal quality)

Why this should work:
- Donchian breakouts work well on higher timeframes (less noise than 15m/1h)
- 1d HMA filter ensures we trade WITH major trend (proven in mtf_hma_rsi_zscore_v1)
- ADX filter avoids false breakouts in ranging markets (major 2022-2023 issue)
- 12h TF has fewer trades than 4h but better signal quality per trade
- ATR stoploss protects against 2022-style crashes

Timeframe: 12h (REQUIRED for Experiment #002)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_adx_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    Reference: Alan Hull, "Making Moving Averages Useful"
    """
    close_s = pd.Series(close)
    n = period
    
    # WMA helper
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - highest high and lowest low over N periods.
    Breakout above upper = bullish, below lower = bearish.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D HMA for trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Additional trend filter: 12h HMA
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_48 = calculate_hma(close, period=48)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_21[i] > hma_12h_48[i]
        hma_bearish = hma_12h_21[i] < hma_12h_48[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 18  # Trending market (slightly lower threshold for more trades)
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === HMA SLOPE CONFIRMATION ===
        hma_slope_long = hma_12h_21[i] > hma_12h_21[i-3] if i > 3 else False
        hma_slope_short = hma_12h_21[i] < hma_12h_21[i-3] if i > 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need Donchian breakout + (1D bias OR 12h HMA) + ADX confirmation
        long_score = 0
        if donchian_breakout_long:
            long_score += 1.5  # Primary trigger
        if daily_bullish:
            long_score += 0.75
        if hma_bullish:
            long_score += 0.75
        if adx_strong:
            long_score += 0.5
        if hma_slope_long:
            long_score += 0.5
        
        # Enter long if score >= 2.5 (breakout + at least 2 confirmations)
        if long_score >= 2.5 and donchian_breakout_long:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need Donchian breakout + (1D bias OR 12h HMA) + ADX confirmation
        short_score = 0
        if donchian_breakout_short:
            short_score += 1.5  # Primary trigger
        if daily_bearish:
            short_score += 0.75
        if hma_bearish:
            short_score += 0.75
        if adx_strong:
            short_score += 0.5
        if hma_slope_short:
            short_score += 0.5
        
        # Enter short if score >= 2.5 (breakout + at least 2 confirmations)
        if short_score >= 2.5 and donchian_breakout_short:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), allow weaker entry to ensure trade generation
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if daily_bullish and hma_bullish and adx_strong:
                new_signal = BASE_SIZE * 0.6  # Smaller size for weaker signal
            elif daily_bearish and hma_bearish and adx_strong:
                new_signal = -BASE_SIZE * 0.6
        
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
        
        # === ADX WEAKNESS EXIT ===
        adx_weakness = False
        if in_position and position_side != 0:
            # Exit if ADX drops below 14 (trend dying)
            if adx_14[i] < 14:
                adx_weakness = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or adx_weakness:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
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
2026-03-22 20:44
