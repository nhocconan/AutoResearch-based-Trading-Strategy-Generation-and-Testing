#!/usr/bin/env python3
"""
Experiment #011: 4h KAMA Adaptive Trend + 1d/1w HMA Filter + ADX Confirmation

Hypothesis: Previous breakout strategies (Donchian, BB Squeeze) failed because they
whipsaw in crypto's noisy ranging periods. This strategy uses ADAPTIVE trend-following:

1. KAMA (Kaufman Adaptive MA) - adapts smoothing based on market efficiency ratio.
   Fast in trends, slow in chop. Period=10, fast=2, slow=30. Proven in crypto.

2. 1d HMA(21) Trend Filter - via mtf_data. Only long if price > 1d HMA, only short
   if price < 1d HMA. Prevents counter-trend trades that destroy Sharpe.

3. 1w HMA(21) Major Bias - via mtf_data. Increases size when 4h+1d+1w align.

4. ADX(14) Trend Strength - ADX > 20 confirms trending market. ADX < 20 = reduce
   position or stay flat. Filters choppy whipsaw periods.

5. Pullback Entry - Enter when price pulls back TO KAMA (not breakout above).
   Long: price crosses ABOVE KAMA from below. Short: price crosses BELOW KAMA from above.
   This catches trend continuations, not false breakouts.

6. RSI(14) Momentum - RSI 45-55 for entry (neutral momentum), RSI > 55 for long
   confirmation, RSI < 45 for short confirmation.

7. ATR(14) Trailing Stop - 2.5x ATR for risk management. Signal → 0 when stopped.

Why this should work:
- KAMA adapts to volatility = fewer whipsaws than fixed EMA/HMA
- ADX filter avoids trading in chop (where 70% of strategies fail)
- Pullback entries have better risk/reward than breakouts
- 1d/1w HTF filters prevent major counter-trend disasters
- 4h timeframe = 20-50 trades/year target (optimal for fee drag)
- Simpler entry logic = more trades (avoids 0-trade failure mode)

Timeframe: 4h (REQUIRED for Experiment #011)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction, 0.20 low conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_pullback_1d_1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    n = period
    
    # Efficiency Ratio = |net change| / sum of absolute changes
    change = np.abs(close_s - close_s.shift(n))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=n, min_periods=n).sum()
    
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing constant
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[n-1] = close_s.iloc[n-1]  # Initialize with first close
    
    for i in range(n, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/choppy.
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
    
    # Smoothed averages (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    adx = adx.replace([np.inf, -np.inf], np.nan)
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, period=10, fast=2, slow=30)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === ADX TREND STRENGTH ===
        trending = adx_14[i] > 20  # ADX > 20 = trending market
        strong_trend = adx_14[i] > 25  # ADX > 25 = strong trend
        
        # === KAMA TREND DIRECTION ===
        kama_rising = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_falling = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # === PRICE VS KAMA ===
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === KAMA CROSSOVER (PULLBACK ENTRY) ===
        kama_cross_long = False
        kama_cross_short = False
        
        if i > 0:
            # Long: price crosses ABOVE KAMA from below
            if price_above_kama and close[i-1] <= kama_10[i-1]:
                kama_cross_long = True
            # Short: price crosses BELOW KAMA from above
            if price_below_kama and close[i-1] >= kama_10[i-1]:
                kama_cross_short = True
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        rsi_neutral = 45 <= rsi_14[i] <= 55
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY - simpler conditions for more trades
        long_score = 0
        
        # Primary: KAMA crossover or price above KAMA
        if kama_cross_long:
            long_score += 3
        elif price_above_kama and kama_rising:
            long_score += 2
        
        # Trend alignment (daily) - REQUIRED for long
        if daily_bullish:
            long_score += 2
        else:
            long_score -= 2  # Penalty for counter-trend
        
        # Major bias (weekly) - adds conviction
        if weekly_bullish:
            long_score += 1
        
        # ADX trend confirmation
        if trending:
            long_score += 1
        if strong_trend:
            long_score += 1
        
        # RSI momentum
        if rsi_strong_bull:
            long_score += 1
        elif rsi_bullish:
            long_score += 0.5
        
        # KAMA direction confirmation
        if kama_rising:
            long_score += 1
        
        # Enter long if score >= 5 (lower threshold for more trades)
        if long_score >= 5:
            # Determine position size based on conviction
            if weekly_bullish and daily_bullish and strong_trend:
                new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction
            elif daily_bullish and trending:
                new_signal = BASE_SIZE  # 0.25 - base
            else:
                new_signal = LOW_CONV_SIZE  # 0.20 - low conviction
        
        # SHORT ENTRY - simpler conditions for more trades
        short_score = 0
        
        # Primary: KAMA crossover or price below KAMA
        if kama_cross_short:
            short_score += 3
        elif price_below_kama and kama_falling:
            short_score += 2
        
        # Trend alignment (daily) - REQUIRED for short
        if daily_bearish:
            short_score += 2
        else:
            short_score -= 2  # Penalty for counter-trend
        
        # Major bias (weekly) - adds conviction
        if weekly_bearish:
            short_score += 1
        
        # ADX trend confirmation
        if trending:
            short_score += 1
        if strong_trend:
            short_score += 1
        
        # RSI momentum
        if rsi_strong_bear:
            short_score += 1
        elif rsi_bearish:
            short_score += 0.5
        
        # KAMA direction confirmation
        if kama_falling:
            short_score += 1
        
        # Enter short if score >= 5
        if short_score >= 5:
            # Determine position size based on conviction
            if weekly_bearish and daily_bearish and strong_trend:
                new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
            elif daily_bearish and trending:
                new_signal = -BASE_SIZE  # -0.25 - base
            else:
                new_signal = -LOW_CONV_SIZE  # -0.20 - low conviction
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~240 hours = 10 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if price_above_kama and daily_bullish and kama_rising:
                new_signal = LOW_CONV_SIZE
            elif price_below_kama and daily_bearish and kama_falling:
                new_signal = -LOW_CONV_SIZE
        
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
            # Exit long if daily trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === KAMA REVERSAL EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Exit long if price falls below KAMA
            if position_side > 0 and price_below_kama:
                kama_exit = True
            # Exit short if price rises above KAMA
            if position_side < 0 and price_above_kama:
                kama_exit = True
        
        # === ADX WEAKNESS EXIT ===
        adx_exit = False
        if in_position and position_side != 0:
            # Exit if ADX drops below 15 (trend dying)
            if adx_14[i] < 15:
                adx_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or kama_exit or adx_exit:
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