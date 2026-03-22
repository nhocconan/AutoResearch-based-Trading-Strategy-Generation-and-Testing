#!/usr/bin/env python3
"""
Experiment #020: 1h Multi-Timeframe Fisher+KAMA with 4h/12h Trend Filter

Hypothesis: Previous 1h strategies failed due to either (a) too many trades → fee drag,
or (b) too strict filters → 0 trades. This strategy balances both by:

1. 4h HMA(21) - Major trend bias via mtf_data helper. Only long if price > 4h HMA,
   only short if price < 4h HMA. Prevents counter-trend trades.

2. 12h ADX(14) - Regime filter. ADX > 20 = trending (follow trend), ADX < 20 = range
   (mean revert). Dual-mode strategy adapts to market conditions.

3. 1h Fisher Transform (period=9) - Entry timing. Long when Fisher crosses above -1.0,
   short when crosses below +1.0. Catches reversals better than RSI.

4. 1h KAMA(10,2,30) - Adaptive trend confirmation. KAMA adapts smoothing based on
   market efficiency. Price > KAMA for longs, < KAMA for shorts.

5. Volume filter - Volume > 0.7x 20-bar average. Avoids low-liquidity traps.

6. RSI(14) filter - RSI < 50 for longs, RSI > 50 for shorts. Simple momentum confirmation.

7. ATR(14) trailing stop - 2.5x ATR for risk management. Signal → 0 when stopped.

Why this should work:
- 4h/12h HTF filters reduce trade frequency to 30-80/year target
- Fisher Transform proven for crypto reversals (better than RSI in 2022 crash)
- KAMA adapts to volatility regimes (works in bull/bear/range)
- Volume filter avoids false breakouts on low liquidity
- 1h timeframe with HTF direction = optimal trade frequency
- Conservative sizing (0.20-0.30) protects against 77% crashes

Timeframe: 1h (REQUIRED for Experiment #020)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Trade target: 30-80/year (120-320 over 4-year train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_4h_12h_adx_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price into near-Gaussian distribution.
    Catches reversals better than RSI, especially in bear markets.
    
    Entry signals:
    - Long: Fisher crosses above -1.0 (oversold reversal)
    - Short: Fisher crosses below +1.0 (overbought reversal)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    n = len(close)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize price over lookback period
    lowest = typical.rolling(window=period, min_periods=period).min()
    highest = typical.rolling(window=period, min_periods=period).max()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, 0.001)
    
    normalized = (typical - lowest) / range_val
    
    # Calculate transform iteratively (needs previous value)
    transform = np.zeros(n)
    fisher = np.zeros(n)
    
    for i in range(period, n):
        # Ehlers smoothing formula
        if i == period:
            transform[i] = 0.66 * ((normalized.iloc[i] - 0.5) + 0.67 * 0)
        else:
            transform[i] = 0.66 * ((normalized.iloc[i] - 0.5) + 0.67 * transform[i-1])
        
        # Clamp to avoid ln domain errors
        transform[i] = np.clip(transform[i], -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + transform[i]) / (1 - transform[i]))
    
    return fisher

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) - adapts to market noise.
    
    Efficiency Ratio (ER) = |close - close(n)| / sum(|close - close(prev)|)
    ER near 1 = trending (use fast SC)
    ER near 0 = choppy (use slow SC)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Price change over ER period
    price_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of absolute price changes (volatility)
    volatility = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = price_change / volatility
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA iteratively
    kama = np.zeros(n)
    kama[er_period] = close_s.iloc[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    n = len(close)
    
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
    
    # Smoothed values (Wilder's smoothing = EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
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
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for major trend bias
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h ADX for regime detection
    adx_12h, _, _ = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1h indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(adx_12h_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(kama[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === 4H MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_4h_21_aligned[i]
        weekly_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 12H REGIME (ADX) ===
        adx_trending = adx_12h_aligned[i] > 20  # Trending regime
        adx_ranging = adx_12h_aligned[i] < 20   # Ranging regime
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama[i] and kama[i] > kama_fast[i]
        kama_bearish = close[i] < kama[i] and kama[i] < kama_fast[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if i > 0 and not np.isnan(fisher[i-1]):
            # Long signal: Fisher crosses above -1.0 from below
            if fisher[i-1] < -1.0 and fisher[i] >= -1.0:
                fisher_cross_up = True
            # Short signal: Fisher crosses below +1.0 from above
            if fisher[i-1] > 1.0 and fisher[i] <= 1.0:
                fisher_cross_down = True
        
        # Fisher extreme levels (for mean reversion in ranging market)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === RSI FILTER ===
        rsi_long_ok = rsi_14[i] < 50
        rsi_short_ok = rsi_14[i] > 50
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY - Need 3+ confluence
        long_score = 0
        long_confidence = 0
        
        # Primary trigger: Fisher reversal
        if fisher_cross_up:
            long_score += 3.0
            long_confidence = 1
        elif fisher_oversold and adx_ranging:
            # Mean reversion in ranging market
            long_score += 2.0
            long_confidence = 0.7
        
        # Trend alignment (4h HMA)
        if weekly_bullish:
            long_score += 1.5
        
        # KAMA confirmation
        if kama_bullish:
            long_score += 1.0
        
        # RSI confirmation
        if rsi_long_ok:
            long_score += 0.5
        
        # Volume confirmation
        if volume_ok:
            long_score += 0.5
        
        # Regime-based boost
        if adx_trending and fisher_cross_up and weekly_bullish:
            long_score += 1.0  # Trending + Fisher cross + trend alignment
        
        # Enter long if score >= 5.0 (strong confluence)
        if long_score >= 5.0:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY - Need 3+ confluence
        short_score = 0
        short_confidence = 0
        
        # Primary trigger: Fisher reversal
        if fisher_cross_down:
            short_score += 3.0
            short_confidence = 1
        elif fisher_overbought and adx_ranging:
            # Mean reversion in ranging market
            short_score += 2.0
            short_confidence = 0.7
        
        # Trend alignment (4h HMA)
        if weekly_bearish:
            short_score += 1.5
        
        # KAMA confirmation
        if kama_bearish:
            short_score += 1.0
        
        # RSI confirmation
        if rsi_short_ok:
            short_score += 0.5
        
        # Volume confirmation
        if volume_ok:
            short_score += 0.5
        
        # Regime-based boost
        if adx_trending and fisher_cross_down and weekly_bearish:
            short_score += 1.0  # Trending + Fisher cross + trend alignment
        
        # Enter short if score >= 5.0 (strong confluence)
        if short_score >= 5.0:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~6 days on 1h), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if fisher_oversold and weekly_bullish and volume_ok:
                new_signal = REDUCED_SIZE
            elif fisher_overbought and weekly_bearish and volume_ok:
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long if Fisher goes overbought (mean reversion complete)
            if position_side > 0 and fisher[i] > 1.0:
                fisher_exit = True
            # Exit short if Fisher goes oversold (mean reversion complete)
            if position_side < 0 and fisher[i] < -1.0:
                fisher_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and weekly_bullish:
                trend_reversal = True
        
        # === KAMA CROSS EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Exit long if price crosses below KAMA
            if position_side > 0 and close[i] < kama[i]:
                kama_exit = True
            # Exit short if price crosses above KAMA
            if position_side < 0 and close[i] > kama[i]:
                kama_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or fisher_exit or trend_reversal or kama_exit:
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