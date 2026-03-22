#!/usr/bin/env python3
"""
Experiment #445: 15m Multi-Confirmation Trend Pullback with 4h HMA Filter

Hypothesis: After 444 failed experiments, the pattern is clear - single-indicator
strategies fail on 15m due to noise. This strategy uses MULTI-CONFIRMATION:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   - HMA smoother than EMA, critical for trend detection

2. 15M HMA(8/21) CROSSOVER for entry timing:
   - Fast HMA(8) crosses above Slow HMA(21) = long trigger
   - Fast HMA(8) crosses below Slow HMA(21) = short trigger
   - HMA has less lag than EMA, better for 15m entries

3. RSI(14) PULLBACK FILTER:
   - Long: RSI between 35-55 (pullback in uptrend, not oversold crash)
   - Short: RSI between 45-65 (rally in downtrend, not overbought spike)
   - Avoids catching falling knives or chasing tops

4. ADX(14) TREND STRENGTH:
   - ADX > 18 = trend strong enough to trade
   - ADX < 18 = skip (choppy market)
   - Lower threshold than traditional 25 to ensure sufficient trades on 15m

5. VOLUME CONFIRMATION:
   - Volume > 1.3 * SMA(volume, 20) = confirmed move
   - Prevents fake breakouts on low volume

6. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 15m noise protection

7. POSITION SIZING: 0.25 discrete (conservative for 15m volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why this should work on 15m:
- 4h HMA filter prevents counter-trend disasters (proven in baseline)
- HMA crossover has less lag than EMA for 15m timing
- RSI pullback filter avoids extreme entries
- Volume confirmation reduces false signals
- ADX > 18 (not 25) ensures sufficient trade frequency (>10/symbol)
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_crossover_4h_hma_rsi_vol_adx_atr_v1"
timeframe = "15m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    values_s = pd.Series(values)
    return values_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    vol_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 15M HMA CROSSOVER ===
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # Also allow continuation (already crossed and holding)
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # === RSI PULLBACK FILTER ===
        rsi_pullback_long = 35 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 65  # Rally in downtrend
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 18  # Lower threshold for 15m
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_sma[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bull + HMA bullish + RSI pullback + ADX strong + volume
        if bull_trend_4h and hma_bullish and rsi_pullback_long and trend_strong:
            if vol_confirmed or hma_cross_long:
                new_signal = SIZE
        
        # SHORT ENTRY: 4h bear + HMA bearish + RSI pullback + ADX strong + volume
        if bear_trend_4h and hma_bearish and rsi_pullback_short and trend_strong:
            if vol_confirmed or hma_cross_short:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === HMA CROSSOVER EXIT (opposite cross) ===
        if in_position and position_side > 0 and hma_cross_short:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_cross_long:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals