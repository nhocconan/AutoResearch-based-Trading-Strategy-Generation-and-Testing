# Strategy: mtf_4h_crsi_hma_chop_12h_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.983 | +3.7% | -6.3% | 389 | FAIL |
| ETHUSDT | -1.689 | -14.3% | -20.8% | 387 | FAIL |
| SOLUSDT | 0.697 | +118.9% | -25.1% | 522 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.114 | +6.6% | -15.9% | 177 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #004: 4h Connors RSI Mean Reversion with 12h/1d Trend Filter

Hypothesis: Previous regime-switching strategies failed because they tried to be
too clever. Research shows Connors RSI (CRSI) has 75% win rate on mean reversion
when combined with trend filter. This strategy uses:

1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - CRSI < 15 = oversold (long entry)
   - CRSI > 85 = overbought (short entry)
2. 12h HMA(21) for trend bias - only long if price > 12h HMA, short if <
3. 1d HMA(21) for major trend filter - avoid counter-trend trades
4. Choppiness Index(14) - only mean revert when CHOP > 55 (ranging market)
5. ATR(14) stoploss - 2.5x ATR trailing stop
6. 4h timeframe - targets 30-50 trades/year (optimal for mean reversion)

Why this should work:
- Connors RSI proven on ETH (Sharpe +0.923 in research)
- 12h/1d HMA filters prevent counter-trend mean reversion (major failure mode)
- Choppiness filter ensures we only mean revert in ranging markets
- 4h TF has enough signals for trade generation but not excessive fees
- Conservative sizing (0.25-0.30) protects against 2022-style crashes

Timeframe: 4h (REQUIRED for Experiment #004)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_hma_chop_12h_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - RSI of consecutive up/down streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss
    streak_rs = streak_rs.replace([np.inf, -np.inf], np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank - percentile of daily change over lookback
    daily_return = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = daily_return.iloc[i-rank_period:i].dropna()
        if len(window) > 0:
            percent_rank[i] = 100 * (daily_return.iloc[i] > window).sum() / len(window)
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
    
    return chop.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for intermediate trend
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Additional 4h trend filter
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H INTERMEDIATE TREND ===
        hma_12h_bullish = close[i] > hma_12h_21_aligned[i]
        hma_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H SHORT-TERM TREND ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55  # Ranging market (mean reversion works)
        chop_trend = chop[i] < 45  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Strong oversold
        crsi_overbought = crsi[i] > 85  # Strong overbought
        crsi_moderate_oversold = crsi[i] < 25  # Moderate oversold
        crsi_moderate_overbought = crsi[i] > 75  # Moderate overbought
        
        # === BOLLINGER BAND POSITION ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: CRSI oversold + trend alignment + ranging market
        long_score = 0
        long_confidence = 0
        
        # Primary trigger: CRSI extreme
        if crsi_oversold:
            long_score += 2.0
            long_confidence = 1
        elif crsi_moderate_oversold and bb_oversold:
            long_score += 1.5
            long_confidence = 0.7
        
        # Trend alignment (need at least 2 of 3 timeframes bullish)
        trend_bullish_count = sum([daily_bullish, hma_12h_bullish, hma_4h_bullish])
        if trend_bullish_count >= 2:
            long_score += 1.5
        elif trend_bullish_count >= 1:
            long_score += 0.75
        
        # Regime filter (mean reversion works best in ranging markets)
        if chop_range:
            long_score += 1.0
        elif not chop_trend:  # Neutral regime
            long_score += 0.5
        
        # Volume confirmation (price near BB lower in range)
        if bb_oversold and chop_range:
            long_score += 0.5
        
        # Enter long if score >= 4.0 (strong confluence)
        if long_score >= 4.0:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY: CRSI overbought + trend alignment + ranging market
        short_score = 0
        short_confidence = 0
        
        # Primary trigger: CRSI extreme
        if crsi_overbought:
            short_score += 2.0
            short_confidence = 1
        elif crsi_moderate_overbought and bb_overbought:
            short_score += 1.5
            short_confidence = 0.7
        
        # Trend alignment (need at least 2 of 3 timeframes bearish)
        trend_bearish_count = sum([daily_bearish, hma_12h_bearish, hma_4h_bearish])
        if trend_bearish_count >= 2:
            short_score += 1.5
        elif trend_bearish_count >= 1:
            short_score += 0.75
        
        # Regime filter
        if chop_range:
            short_score += 1.0
        elif not chop_trend:
            short_score += 0.5
        
        # Volume confirmation
        if bb_overbought and chop_range:
            short_score += 0.5
        
        # Enter short if score >= 4.0 (strong confluence)
        if short_score >= 4.0:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if crsi_moderate_oversold and trend_bullish_count >= 2:
                new_signal = REDUCED_SIZE
            elif crsi_moderate_overbought and trend_bearish_count >= 2:
                new_signal = -REDUCED_SIZE
        
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long if CRSI goes overbought (mean reversion complete)
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short if CRSI goes oversold (mean reversion complete)
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if major trend turns bearish
            if position_side > 0 and daily_bearish and hma_12h_bearish:
                trend_reversal = True
            # Exit short if major trend turns bullish
            if position_side < 0 and daily_bullish and hma_12h_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or crsi_exit or trend_reversal:
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
2026-03-22 20:46
