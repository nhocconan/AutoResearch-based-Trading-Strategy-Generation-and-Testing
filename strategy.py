#!/usr/bin/env python3
"""
Experiment #013: 1d Weekly Trend Filter with Daily RSI Pullback + Volatility Sizing

Hypothesis: Previous 1d strategies failed because they were either too complex (regime-switching
with many filters = 0 trades) or used pure trend-following (whipsaw in 2022). This strategy uses:

1. 1w HMA(21) for MAJOR trend bias - only long if price > weekly HMA, short if <
   This is the proven edge from mtf_hma_rsi_zscore_v1 (Sharpe=5.4)
2. 1d RSI(14) for entry timing - enter on pullbacks in trend direction
   Long: RSI < 40 in weekly uptrend | Short: RSI > 60 in weekly downtrend
3. ADX(14) filter - only trade when ADX > 20 (avoid choppy whipsaw)
4. ATR(14) volatility-adjusted position sizing - reduce size when vol is high
5. 2.5 ATR trailing stoploss - protects against 2022-style crashes
6. 1d timeframe - targets 20-40 trades/year (optimal for swing trading)

Why this should work:
- Weekly trend filter prevents counter-trend trades (major failure mode in 2022)
- RSI pullback entries have higher win rate than breakouts in crypto
- ADX filter avoids ranging market whipsaw
- Volatility sizing reduces exposure during high-vol periods (crashes)
- Simple logic = more trades generated (avoids 0-trade failure)
- 1d TF has lower fee drag than lower timeframes

Timeframe: 1d (REQUIRED for Experiment #013)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 with vol adjustment
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_rsi_adx_vol_v1"
timeframe = "1d"
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

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index) - measures trend strength.
    ADX > 25 = trending | ADX < 20 = ranging
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
    
    # Smoothed values using Wilder's method (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volatility_ratio(atr, period=30):
    """
    Calculate volatility ratio = ATR / rolling mean ATR
    Used for position sizing - reduce size when vol is high
    """
    atr_s = pd.Series(atr)
    atr_mean = atr_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = atr_s / atr_mean
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    rsi_14 = calculate_rsi(close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volatility_ratio(atr_14, period=30)
    
    # Additional 1d trend filter
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    MIN_SIZE = 0.12
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            vol_ratio[i] = 1.0  # Default to normal vol
        
        # === 1W MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        daily_bullish = hma_1d_21[i] > hma_1d_50[i]
        daily_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Trending market
        adx_weak = adx_14[i] < 15    # Ranging market (avoid)
        
        # === RSI ENTRY LEVELS ===
        rsi_oversold = rsi_14[i] < 40   # Pullback in uptrend
        rsi_overbought = rsi_14[i] > 60  # Pullback in downtrend
        rsi_extreme_oversold = rsi_14[i] < 30
        rsi_extreme_overbought = rsi_14[i] > 70
        
        # === VOLATILITY-ADJUSTED POSITION SIZE ===
        # Reduce size when volatility is high (protect against crashes)
        if vol_ratio[i] > 1.5:
            position_size = MIN_SIZE
        elif vol_ratio[i] > 1.2:
            position_size = REDUCED_SIZE
        else:
            position_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Weekly uptrend + RSI pullback + ADX confirms trend
        if weekly_bullish and rsi_oversold and adx_strong:
            # Strong entry: extreme RSI + both timeframes bullish
            if rsi_extreme_oversold and daily_bullish:
                new_signal = position_size
            # Moderate entry: normal RSI pullback
            elif daily_bullish:
                new_signal = REDUCED_SIZE
            # Weak entry: weekly bullish but daily neutral
            elif bars_since_last_trade > 60:
                new_signal = MIN_SIZE
        
        # SHORT ENTRY: Weekly downtrend + RSI pullback + ADX confirms trend
        elif weekly_bearish and rsi_overbought and adx_strong:
            # Strong entry: extreme RSI + both timeframes bearish
            if rsi_extreme_overbought and daily_bearish:
                new_signal = -position_size
            # Moderate entry: normal RSI pullback
            elif daily_bearish:
                new_signal = -REDUCED_SIZE
            # Weak entry: weekly bearish but daily neutral
            elif bars_since_last_trade > 60:
                new_signal = -MIN_SIZE
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI goes overbought (pullback complete)
            if position_side > 0 and rsi_14[i] > 65:
                rsi_exit = True
            # Exit short if RSI goes oversold (pullback complete)
            if position_side < 0 and rsi_14[i] < 35:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish:
                trend_reversal = True
        
        # === ADX WEAKNESS EXIT ===
        adx_exit = False
        if in_position and position_side != 0:
            # Exit if trend strength disappears (ADX < 15)
            if adx_weak:
                adx_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or rsi_exit or trend_reversal or adx_exit:
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